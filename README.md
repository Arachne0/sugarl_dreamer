## ⚠️ Warning ⚠️

그냥 clone해서 시작하지 마시고, 이 README.md 전부 읽어보시고 시작해주세요.
이해가 안되신다면 연락 부탁드려요.

<br>


###  Working Directory 📁
git clone 했을 때, 여러 개 폴더가 보이실 텐데 dreamerv3-torch 폴더를 제외한 나머지 폴더는 전부 SUGARL 에서 가져온 거 입니다.

저희가 main으로 해야할 게  "SUGARL에서 Sensory policy 부분을 DreamerV3로 교체하자" 입니다. 

main working directory는 agent_dreamer 폴더이며, agent 는 기존에 SUGARL에 있던 파일들입니다. 

일단 Atari에서 SAC를 사용했을 때, SUGARL + DreamerV3 가 잘 되는지를 보고자 하는 것이며, 추후에 DMC (2D, 3D로 확장할 예정입니다.)

<br>

### Current Progress (10/1)
1. Dreamer network issue

원본 코드에서는 image size를 64로 설정했었고, 항상 똑같은 사이즈의 필터 사이즈와 패딩이 적용되도록 했는데 (4장) 우리는 SUGARL의 이미지 사이즈에서 비교해야하기 때문에 84, 84로 수정하는 과정에서 network shape이 맞지 않는 문제가 있었다.

이걸 SUGARL에서의 network를 떼와서 world model의 인코더와 디코더에 붙였다. network 사이즈와 패딩만 없는 그런 차이점만 있을 뿐 다른 차이점은 최대한 동일하게 유지했다.

2. Dreamer image shape issue

원본 Dreamer v3 코드에서는 stack해서 하지 않고 RGB로 했었는데(64, 64, 3), SUGARL에서는 이를 stack해서 쓰기 때문에 Dreamer에서도 stack했을 때 제대로 작동하는 지를 확인할 필요가 있었다. envs/wrappers.py 에다가 Framestack 클래스를 정의하고 wrappers로 감쌌을 때 shape 잘 들어가는 거 확인했다.

3. Future works

이제 SUGARL 에다가 dreamer 합치기만 하면 된다.

<br>

### Must be fixed 🛠️ 

1. 각각의 gym 버전이 다를 가능성이 매우 높다.
잠깐 보기엔 active vision RL에서는 gymnasium as gym (version=0.29.1)해서 쓰고, 
dreamerv3에서는 그냥 gym (version==0.19.0)을 쓴다.  맞는 버전이 있다면 좋겠다만 맞지 않더라도 잘 수정해야한다.
최대한 잘 정리해서 최대한 딸깍 몇번으로 interpreter 구성할 수 있게 만들어보자/

2. 이젠 진짜 2개 모델 합쳐야 한다.


3. dreamer.py 디버깅 중에  지금은 config.task 를 "atari_pong" 으로 configs.yaml에서 고정시켜서 아마 이것만 할텐데 최종적으로는 atari에 대한 모든 환경을 config.task에 자동적으로 넣어서 돌아갈 수 있도록 만들어야 한다. 



## 🌱  Working Branch 🌱 

```
git checkout -b <your_branch_name>
```
밑의 예시처럼 각각의 branch를 만들고 push할 때, 어느정도 진행되어서 정리 해둘 필요 있는거 아닌 이상 main에다가 push 절대 하지 마세요.

```
git push origin main 
```
이거 했다가 나중에 version 꼬이면 상당히 골치 아파요.

```
git push origin <your_branch_name>
```
이렇게 부탁드립니다.

<br>



##  Dependency 📦

Repo 를 2개를 임의로 붙인거다 보니 중간에 호환안되거나 문제 생기는 점은 update하겠습니다. python 버전은 3.9 입니다. 

- active RL
```bash
conda env create -f active_rl_env.yaml
conda activate arl 
pip install gymnasium[accept-rom-license]
AutoROM --accept-license
```

- Dreamer
```bash
sudo apt-get update
sudo apt-get install unrar
bash dreamerv3-torch/envs/setup_scripts/atari.sh 
pip install "gym[atari,accept-rom-license]"
```

- Dreamer gym problem solved
```bash
pip install "pip<24.1"
pip install "gym==0.19.0"
sudo apt-get update
sudo apt-get install build-essential cmake
conda update libstdcxx-ng
conda install gxx_linux-64
mv /home/hail/anaconda3/envs/dreamer/lib/libstdc++.so.6 /home/hail/anaconda3/envs/dreamer/lib/libstdc++.so.6.bak
pip install gym[atari]
```

<br>

⚠️ 그대로 yaml 파일 적용시 ROM 경로가 다른데 들어가서 복사해주는 작업이 필요합니다. 밑의 코드를 그대로 복사하면 절대 경로라 에러 발생할 가능성이 매우 높은이 anaconda3 경로 위치 확인하고 하세요. ⚠️

```bash
python -m atari_py.import_roms /home/hail/anaconda3/envs/arl/lib/python3.9/site-packages/AutoROM/roms
```

<br>

### Notes

`scripts/` 안에 있는 scripts는 병렬 처리가 가능하도록, SURAGL 저자들이 shell script로 작성해둔 것입니다.

여기에서 Atari만 해도 환경을 26개를 넣어놨는데, 4090에서 3번 돌렸다가 전부 죽어버려서 `agent_dreamer/` 이 폴더에서 agent를 반복문으로 집어넣어주는 식으로 변경하였습니다. 다만 `scripts/` 안에 있는 몇몇 arguments 들이 필요할 수 있어 혹시나 남겨두었습니다.

main branch로 올라간 첫 버전은 그냥 SUGARL를 wandb에 찍을 수 있도록 몇개의 줄을 추가한 게 다이며, 추가 수정 사항 있을 시 README나 다른 수단으로 공유하도록 하겠습니다.



- 09/12 (Fri) Donggyu Lab meeting 

[PDF Download]({% raw %}_docs{% endraw %}/Active%20Vision%20Reinforcement%20Learning%20under%20Limited%20Vis...pdf)


<br>


## agent_dreamer launch.json arguments
- sac_atari_sugarl_dreamer_50x50.py 
```
"args": ["--seed", "0",
        "--exp-name", "atari_100k_50x50",
        "--fov-size", "50",
        "--clip-reward",
        "--capture-video",
        "--total-timesteps", "1000000",
        "--buffer-size", "100000",
        "--learning-starts", "80000"
    ],
```

<br>

## DreamerV3
dreamer v3는 자동적으로 configs.yaml을 읽어오기 때문에 argument 따로 안넣어줘도 됩니다.
그냥 디버깅 해도 잘 돌아가요


<br>

## Citation
Please consider cite us if you find this repo helpful.

- Active RL 
```
@article{shang2023active,
    title={Active Reinforcement Learning under Limited Visual Observability},
    author={Jinghuan Shang and Michael S. Ryoo},
    journal={arXiv preprint},
    year={2023},
    eprint={2306.00975},
}
```

- Dreamer v3
```
@article{hafner2023mastering,
  title={Mastering diverse domains through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```