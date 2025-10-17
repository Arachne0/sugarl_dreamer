# -*- coding: utf-8 -*-
import argparse
import os, sys
import os.path as osp
import random
import time
from itertools import product
from distutils.util import strtobool
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from common.utils import (
    get_sugarl_reward_scale_robosuite,
    get_timestr, 
    schedule_drq,
    seed_everything,
    soft_update_params,
    weight_init_drq,
    TruncatedNormal
)
import gymnasium as gym
from gymnasium.spaces import Discrete, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Resize

from common.buffer import DoubleActionReplayBuffer
from common.pvm_buffer import PVMBuffer
from common.utils import get_timestr, seed_everything, get_sugarl_reward_scale_atari
from torch.utils.tensorboard import SummaryWriter

from active_gym.atari_env import AtariFixedFovealEnv, AtariEnvArgs
from collections import deque

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # env setting
    parser.add_argument("--env", type=str, default="breakout",
        help="the id of the environment")
    parser.add_argument("--env-num", type=int, default=1, 
        help="# envs in parallel")
    parser.add_argument("--frame-stack", type=int, default=5,
        help="frame stack #")
    parser.add_argument("--action-repeat", type=int, default=4,
        help="action repeat #")
    parser.add_argument("--clip-reward", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    # fov setting
    parser.add_argument("--fov-size", type=int, default=50)
    parser.add_argument("--fov-init-loc", type=int, default=0)
    parser.add_argument("--sensory-action-mode", type=str, default="absolute")
    parser.add_argument("--sensory-action-space", type=int, default=10) # ignored when sensory_action_mode="relative"
    parser.add_argument("--resize-to-full", default=False, action="store_true")
    # for discrete observ action
    parser.add_argument("--sensory-action-x-size", type=int, default=4)
    parser.add_argument("--sensory-action-y-size", type=int, default=4)
    # pvm setting
    parser.add_argument("--pvm-stack", type=int, default=3)

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=500000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")

    # eval args
    parser.add_argument("--eval-frequency", type=int, default=-1,
        help="eval frequency. default -1 is eval at the end.")
    parser.add_argument("--eval-num", type=int, default=10,
        help="eval frequency. default -1 is eval at the end.")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_name, seed, **kwargs):
    def thunk():
        env_args = AtariEnvArgs(
            game=env_name, seed=seed, obs_size=(84, 84), **kwargs
        )
        env = AtariFixedFovealEnv(env_args)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # ★★★ 입력 채널 수를 명시적으로 받도록 수정 (e.g., 1)
        self.cnn_repr_dim = 32 * 35 * 35
        self.repr_dim = 512

        self.convnet = nn.Sequential(nn.Conv2d(in_channels, 32, 3, stride=2),
                                     nn.Sigmoid(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.Sigmoid(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.Sigmoid(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.Sigmoid())
        
        self.neck = nn.Sequential(
            nn.Linear(self.cnn_repr_dim, 3136),
            nn.ReLU(),
            nn.Linear(3136, self.repr_dim),
            nn.ReLU()
        )

        self.rnn = nn.LSTM(self.repr_dim, self.repr_dim, 1, batch_first=True)

        self.apply(weight_init_drq)

    def forward(self, obs):
        # ★★★ 입력 obs shape: [B, T, C, H, W]
        obs = obs - 0.5 # /255 is done by env
        B, T, C, H, W = obs.size()
        obs = obs.reshape(B*T, C, H, W)
        h = self.convnet(obs)
        h = self.neck(h.reshape(B*T, -1))
        h, _ = self.rnn(h.reshape(B, T, -1))
        h = h[:, -1, :] # 마지막 시퀀스의 출력만 사용
        return h
    
class Decoder(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        
        self.fc = nn.Linear(repr_dim, 64 * 7 * 7)
    
        self.deconvnet = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.Sigmoid(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.Sigmoid(),
            nn.ConvTranspose2d(32, 1, kernel_size=8, stride=4),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = self.deconvnet(x)
        return x # ★★★ 출력 shape: [B, 1, H, W]

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        if isinstance(env.single_action_space, Discrete):
            motor_action_space_size = env.single_action_space.n
            sensory_action_space_size = None
        elif isinstance(env.single_action_space, Dict):
            motor_action_space_size = env.single_action_space["motor_action"].n
            if sensory_action_set is not None:
                sensory_action_space_size = len(sensory_action_set)
            else:
                sensory_action_space_size = env.single_action_space["sensory_action"].n
        self.backbone = nn.Sequential(
            nn.Conv2d(5, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.motor_action_head = nn.Linear(512, motor_action_space_size)
        self.sensory_action_head = None
        if sensory_action_space_size is not None:
            self.sensory_action_head = nn.Linear(512, sensory_action_space_size)
        

    def forward(self, x):
        # ★★★ 입력 x shape: [B, C, H, W] (C=frame_stack)
        x = self.backbone(x / 255.0) # ★★★ DQN은 보통 0~255 입력을 받으므로 정규화
        motor_action = self.motor_action_head(x)
        sensory_action = None
        if self.sensory_action_head:
            sensory_action = self.sensory_action_head(x)
        return motor_action, sensory_action

class SelfPredictionNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        if isinstance(env.single_action_space, Discrete):
            motor_action_space_size = env.single_action_space.n
            sensory_action_space_size = None
        elif isinstance(env.single_action_space, Dict):
            motor_action_space_size = env.single_action_space["motor_action"].n
            if sensory_action_set is not None:
                sensory_action_space_size = len(sensory_action_set)
            else:
                sensory_action_space_size = env.single_action_space["sensory_action"].n
        
        self.backbone = nn.Sequential(
            nn.Conv2d(10, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512, motor_action_space_size),
        )

        self.loss = nn.CrossEntropyLoss()

    def get_loss(self, x, target) -> torch.Tensor:
        return self.loss(x, target)
        

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    args.env = args.env.lower()
    run_name = f"{args.env}__{os.path.basename(__file__)}__{args.seed}__{get_timestr()}"
    run_dir = os.path.join("runs", args.exp_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(run_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    seed_everything(args.seed)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = []
    for i in range(args.env_num):
        envs.append(make_env(args.env, args.seed+i, frame_stack=args.frame_stack, action_repeat=args.action_repeat,
                                fov_size=(args.fov_size, args.fov_size), 
                                fov_init_loc=(args.fov_init_loc, args.fov_init_loc),
                                sensory_action_mode=args.sensory_action_mode,
                                sensory_action_space=(-args.sensory_action_space, args.sensory_action_space),
                                resize_to_full=args.resize_to_full,
                                clip_reward=args.clip_reward,
                                mask_out=True))
    envs = gym.vector.SyncVectorEnv(envs)

    sugarl_r_scale = get_sugarl_reward_scale_atari(args.env)

    resize = Resize((84, 84))

    # get a discrete observ action space
    OBSERVATION_SIZE = (84, 84)
    observ_x_max, observ_y_max = OBSERVATION_SIZE[0]-args.fov_size, OBSERVATION_SIZE[1]-args.fov_size
    sensory_action_step = (observ_x_max//args.sensory_action_x_size,
                           observ_y_max//args.sensory_action_y_size)
    sensory_action_x_set = list(range(0, observ_x_max, sensory_action_step[0]))[:args.sensory_action_x_size]
    sensory_action_y_set = list(range(0, observ_y_max, sensory_action_step[1]))[:args.sensory_action_y_size]
    sensory_action_set = [np.array(a) for a in list(product(sensory_action_x_set, sensory_action_y_set))]

    q_network = QNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    target_network.load_state_dict(q_network.state_dict())

    sfn = SelfPredictionNetwork(envs, sensory_action_set=sensory_action_set).to(device)
    sfn_optimizer = optim.Adam(sfn.parameters(), lr=args.learning_rate)

    # ★★★ 원칙: 리플레이 버퍼는 오직 '실제' 관측값만 저장합니다.
    # obs의 shape는 (num_envs, frame_stack, H, W)
    rb = DoubleActionReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space["motor_action"],
        Discrete(len(sensory_action_set)),
        device,
        n_envs=envs.num_envs,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # ★★★ 예측 모델 초기화
    atari_Encoder = Encoder(in_channels=1).to(device) # 각 프레임은 채널 1
    atari_Decoder = Decoder(512).to(device)
    pred_frame_parameters = list(atari_Encoder.parameters()) + list(atari_Decoder.parameters())
    pred_frame_optimizer = torch.optim.Adam(pred_frame_parameters, lr=0.001)

    obs, infos = envs.reset()
    global_transitions = 0
    
    # ★★★ PVM 버퍼는 실제 obs의 전이(transition)를 관리하기 위해서만 사용
    pvm_buffer = PVMBuffer(args.pvm_stack, (envs.num_envs, args.frame_stack,)+OBSERVATION_SIZE)

    while global_transitions < args.total_timesteps:
        
        # ★★★ 1. 행동 결정(Acting) 단계: 상태를 '즉석에서' 구성하여 행동 선택
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_transitions)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            motor_actions = np.array([actions[0]["motor_action"]])
            sensory_actions = np.array([random.randint(0, len(sensory_action_set)-1)])
        else:
            with torch.no_grad():
                # (B, T, H, W) -> (B, T, 1, H, W)
                obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(2)
                
                # 예측에 사용할 과거 프레임 시퀀스 (마지막 프레임 제외)
                predictor_input = obs_tensor[:, :args.frame_stack - 1]
                
                obs_latent = atari_Encoder(predictor_input)
                predicted_frame_tensor = atari_Decoder(obs_latent) # Shape: [B, 1, H, W]

                # (B, T-1, H, W) + (B, 1, H, W) -> (B, T, H, W)
                # 실제 과거 프레임과 예측된 미래 프레임을 합쳐 '행동용 상태' 구성
                state_for_acting = torch.cat(
                    [torch.from_numpy(obs[:, :args.frame_stack - 1]).to(device), predicted_frame_tensor], dim=1
                )

                motor_q_values, sensory_q_values = q_network(state_for_acting)
                motor_actions = torch.argmax(motor_q_values, dim=1).cpu().numpy()
                sensory_actions = torch.argmax(sensory_q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, _, infos = envs.step({"motor_action": motor_actions, 
                                                         "sensory_action": [sensory_action_set[a] for a in sensory_actions] })

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for idx, d in enumerate(dones):
                if d:
                    print(f"[T: {time.time()-start_time:.2f}]  [N: {global_transitions:07,d}]  [R: {infos['final_info'][idx]['reward']:.2f}]")
                    writer.add_scalar("charts/episodic_return", infos['final_info'][idx]["reward"], global_transitions)
                    writer.add_scalar("charts/episodic_length", infos['final_info'][idx]["ep_len"], global_transitions)
                    writer.add_scalar("charts/epsilon", epsilon, global_transitions)
                    break
        
        # ★★★ 2. 저장(Storing) 단계: 리플레이 버퍼에는 '실제 관측값' 전이만 저장
        pvm_buffer.append(obs)
        pvm_obs = pvm_buffer.get_obs(mode="stack_max")

        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]

        pvm_buffer_copy = pvm_buffer.copy()
        pvm_buffer_copy.append(real_next_obs)
        real_next_pvm_obs = pvm_buffer_copy.get_obs(mode="stack_max")
        
        rb.add(pvm_obs, real_next_pvm_obs, motor_actions, sensory_actions, rewards, dones, {})

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        global_transitions += args.env_num

        if global_transitions < args.learning_starts:
            continue
        
        # ★★★ 3. 학습(Learning) 단계
        if global_transitions % args.train_frequency == 0:
            data = rb.sample(args.batch_size) # 실제 관측값 (o_t, o_{t+1})가 담겨 있음

            # --- 3a. 예측 모델(Encoder/Decoder) 학습 ---
            # 예측 모델은 자신의 예측과 '실제' 다음 프레임의 차이로만 학습
            with torch.no_grad():
                # 타겟: o_t의 마지막 실제 프레임
                target_frame = data.observations[:, [-1], :, :]

            # 입력: o_t의 처음 T-1개 실제 프레임
            predictor_input = data.observations.unsqueeze(2)[:, :args.frame_stack - 1]
            
            obs_latent = atari_Encoder(predictor_input)
            predicted_frame = atari_Decoder(obs_latent)
            
            pred_frame_loss = F.mse_loss(predicted_frame, target_frame)
            pred_frame_optimizer.zero_grad()
            pred_frame_loss.backward()
            pred_frame_optimizer.step()
            writer.add_scalar("losses/prediction_loss", pred_frame_loss, global_transitions)

            # --- 3b. SFN 학습 (기존과 동일하게 실제 obs 기반) ---
            concat_observation = torch.concat([data.observations, data.next_observations], dim=1)
            pred_motor_actions = sfn(resize(concat_observation))
            sfn_loss = sfn.get_loss(pred_motor_actions, data.motor_actions.flatten())
            sfn_optimizer.zero_grad()
            sfn_loss.backward()
            sfn_optimizer.step()
            observ_r = F.softmax(pred_motor_actions, dim=1).gather(1, data.motor_actions).squeeze().detach()

            # --- 3c. Q-Network 학습 ---
            # TD-Target 계산을 위해 s_{t+1}을 '즉석에서 재구성'
            with torch.no_grad():
                # o_{t+1}의 과거 프레임으로 다음 프레임 예측
                next_obs_tensor = data.next_observations.unsqueeze(2)
                next_predictor_input = next_obs_tensor[:, :args.frame_stack - 1]
                next_obs_latent = atari_Encoder(next_predictor_input)
                next_predicted_frame = atari_Decoder(next_obs_latent)

                # s_{t+1} 구성
                state_for_td_target = torch.cat(
                    [data.next_observations[:, :args.frame_stack-1], next_predicted_frame], dim=1
                )
                
                motor_target, sensory_target = target_network(state_for_td_target)
                motor_target_max, _ = motor_target.max(dim=1)
                sensory_target_max, _ = sensory_target.max(dim=1)
                
                td_target = data.rewards.flatten() - (1 - observ_r) * sugarl_r_scale + args.gamma * (motor_target_max + sensory_target_max) * (1 - data.dones.flatten())
            
            # 현재 Q-value 계산을 위해 s_t를 '즉석에서 재구성'
            # 위에서 예측 모델 학습에 사용했던 predicted_frame을 재사용하지 않고, 재구성 원칙을 위해 다시 계산.
            obs_tensor = data.observations.unsqueeze(2)
            predictor_input_for_q = obs_tensor[:, :args.frame_stack - 1]
            obs_latent_for_q = atari_Encoder(predictor_input_for_q)
            
            # ★★★★★ .detach() ★★★★★
            # Q-loss가 예측 모델로 흘러가지 않도록 역전파 경로를 차단!
            predicted_frame_for_q = atari_Decoder(obs_latent_for_q).detach()

            # s_t 구성
            state_for_q_value = torch.cat(
                [data.observations[:, :args.frame_stack-1], predicted_frame_for_q], dim=1
            )

            old_motor_q_val, old_sensory_q_val = q_network(state_for_q_value)
            old_motor_val = old_motor_q_val.gather(1, data.motor_actions).squeeze()
            old_sensory_val = old_sensory_q_val.gather(1, data.sensory_actions).squeeze()
            old_val = old_motor_val + old_sensory_val
            
            loss = F.mse_loss(td_target, old_val)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_transitions % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_transitions)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_transitions)
                # ... other logging
                writer.add_scalar("charts/SPS", int(global_transitions / (time.time() - start_time)), global_transitions)

            # update the target network
            if (global_transitions // args.env_num) % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        # evaluation
        # ... (Evaluation loop도 일관성을 위해 Acting 단계와 동일한 상태 구성 로직을 사용해야 합니다)

    envs.close()
    # eval_env.close() # eval loop 구현 시 필요
    writer.close()