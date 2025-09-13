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



### Must be fixed 🛠️ 

제가 main branch의 dreamerv3-torch/dreamer.py 파일에서 
configs_path = Path("/home/hail/SH/sugarl/dreamerv3-torch/configs.yaml")

이 줄에서 절대 경로로 넣어서 configs_path를 여러분들의 코드 위치로 변경해주셔야 합니다. 추후에 수정하겠지만 아마 main branch가 update 되지 않았을 수 있어 양해부탁드립니다.


<br>



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
굳이 버전 입력 안해도 yaml 안에 있는 python 3.9 로 알아서 설정됩니다.


```
conda env create -f active_rl_env.yaml
conda activate arl 

pip install gymnasium[accept-rom-license]
AutoROM --accept-license
```

<br>

⚠️ 그대로 yaml 파일 적용시 ROM 경로가 다른데 들어가서 복사해주는 작업이 필요합니다. 밑의 코드를 그대로 복사하면 절대 경로라 에러 발생할 가능성이 매우 높은이 anaconda3 경로 위치 확인하고 하세요. ⚠️
```
python -m atari_py.import_roms /home/hail/anaconda3/envs/arl/lib/python3.9/site-packages/AutoROM/roms
```

<br>

⚠️ dreamerv3-torch/active_rl_env.yaml 들어가서 맨 윗 줄에 name을 arl로 (아마 위의 설명 그대로 했다면 conda name은 arl 일 것. ) python 은 3.9로 수정 후 밑의 줄 실행하세요. ⚠️
```
cd dreamerv3-torch
conda env create -f active_rl_env.yaml
```

<br>

### Notes

`scripts/` 안에 있는 scripts는 병렬 처리가 가능하도록, SURAGL 저자들이 shell script로 작성해둔 것입니다.

여기에서 Atari만 해도 환경을 26개를 넣어놨는데, 4090에서 3번 돌렸다가 전부 죽어버려서 `agent_dreamer/` 이 폴더에서 agent를 반복문으로 집어넣어주는 식으로 변경하였습니다. 다만 `scripts/` 안에 있는 몇몇 arguments 들이 필요할 수 있어 혹시나 남겨두었습니다.

main branch로 올라간 첫 버전은 그냥 SUGARL를 wandb에 찍을 수 있도록 몇개의 줄을 추가한 게 다이며, 추가 수정 사항 있을 시 README나 다른 수단으로 공유하도록 하겠습니다.



- 09/12 (Fri) Donggyu Lab meeting 

[PDF Download](_docs/Active Vision Reinforcement Learning under Limited Visual Observability_.pdf)


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


## Citation
Please consider cite us if you find this repo helpful.
```
@article{shang2023active,
    title={Active Reinforcement Learning under Limited Visual Observability},
    author={Jinghuan Shang and Michael S. Ryoo},
    journal={arXiv preprint},
    year={2023},
    eprint={2306.00975},
}
```
