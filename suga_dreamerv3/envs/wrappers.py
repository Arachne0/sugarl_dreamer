import datetime
import gym
import numpy as np
import uuid
from collections import deque
# from gym.spaces import Dict, Box, Discrete
from gymnasium.spaces import Dict, Box, Discrete


class TimeLimit(gym.Wrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        # state, return_reward, done, truncated, info
        obs, return_reward, done, truncated, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        obs = {"image":obs, "is_terminal": False, "is_first": True}
        return obs, return_reward, done, truncated, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, Dict), "This wrapper only works for Dict action spaces."
        assert 'motor_action' in env.action_space.spaces and isinstance(env.action_space.spaces['motor_action'], Discrete)
        
        self._random = np.random.RandomState()
        motor_space = env.action_space.spaces['motor_action']
        sensory_space = env.action_space.spaces['sensory_action']
        motor_space.discrete = True
        
        action_space = Dict({
            'motor_action': motor_space,
            'sensory_action': sensory_space  
        })

        self.action_space = action_space

    def step(self, action):
        action = action.item()
        motor_action_onehot = action['motor_action']
        motor_action_onehot = motor_action_onehot[0].numpy()
        motor_action_index = np.argmax(motor_action_onehot).astype(int)
        reference = np.zeros_like(motor_action_onehot)
        reference[motor_action_index] = 1
        
        if not np.allclose(reference, motor_action_onehot):
            raise ValueError(f"Invalid one-hot motor_action:\n{motor_action_onehot}")
        
        env_action = {
            'motor_action': motor_action_index,
            'sensory_action': action['sensory_action']
        }
        
        return self.env.step(env_action)
    
        # index = np.argmax(action).astype(int)
        # reference = np.zeros_like(action)
        # reference[index] = 1
        # if not np.allclose(reference, action):
        #     raise ValueError(f"Invalid one-hot action:\n{action}")
        # return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        # motor action sampling
        motor_actions_n = self.env.action_space.spaces['motor_action'].n
        motor_index = self._random.randint(0, motor_actions_n)
        reference_motor = np.zeros(motor_actions_n, dtype=np.float32)
        reference_motor[motor_index] = 1.0
        
        # sensory action sampling
        reference_sensory = self.env.action_space.spaces['sensory_action'].sample()
        
        return {
            'motor_action': reference_motor,
            'sensory_action': reference_sensory
        }
        
        # actions = self.env.action_space.n
        # index = self._random.randint(0, actions)
        # reference = np.zeros(actions, dtype=np.float32)
        # reference[index] = 1.0
        # return reference


class RewardObs(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        spaces = self.env.observation_space.spaces
        if "obs_reward" not in spaces:
            spaces["obs_reward"] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()


# class FrameStack(gym.Wrapper):
#     def __init__(self, env, num_frames):
#         super().__init__(env)
        
#         self.num_frames = num_frames
        
#         # Check if the observation is a dictionary and contains the 'image' key
#         if not isinstance(env.observation_space, gym.spaces.Dict) or 'image' not in env.observation_space.spaces:
#             raise ValueError(
#                 "FrameStack wrapper requires a Dict observation space with an 'image' key."
#             )
            
#         image_space = env.observation_space['image']
#         self.shape = image_space.shape
#         self.dtype = image_space.dtype
        
#         # Initialize the deque (buffer) to store the last N frames
#         self._frames = deque(maxlen=num_frames)
        
#         # 1. Update the observation space
#         # New shape will be (H, W, C * num_frames)
#         new_shape = self.shape[:-1] + (self.shape[-1] * num_frames,)
        
#         # 2. Re-create the observation space (keeping other keys as is)
#         new_obs_space = env.observation_space.spaces.copy()
#         new_obs_space['image'] = gym.spaces.Box(
#             low=0, high=255, shape=new_shape, dtype=self.dtype
#         )
#         self._observation_space = gym.spaces.Dict(new_obs_space)

#     def _get_observation(self):
#         # Ensure the number of frames in the deque matches num_frames
#         assert len(self._frames) == self.num_frames
    
#         return np.concatenate(list(self._frames), axis=-1)

#     def reset(self, **kwargs):
#         self._step = 0
    
#         obs_dict  = self.env.reset(**kwargs)
        
#         initial_frame = obs_dict['image']
#         self._frames.clear()
#         for _ in range(self.num_frames):
#             self._frames.append(initial_frame.copy()) 
            
#         stacked_obs = self._get_observation()
#         obs_dict['image'] = stacked_obs
    
#         return obs_dict, {} 

#     def step(self, action):
#         obs_dict, reward, done, info = self.env.step(action) 
        
#         # Get the current frame and push it to the buffer
#         current_frame = obs_dict['image']
#         self._frames.append(current_frame)
#         stacked_obs = self._get_observation()
        
#         # IMPORTANT: Update the 'image' key and preserve all other metadata
#         # (e.g., is_first, is_terminal, if they are in obs_dict)
#         obs_dict['image'] = stacked_obs

#         return obs_dict, reward, done, info