##  Dependency 📦

```bash
conda create -n suga_dreamer python=3.9 -y
conda activate suga_dreamer
pip install "pip<24.1" "setuptools<60.0.0" "wheel<0.40.0"
pip install -r suga_dreamerv3/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116

AutoROM --accept-license
python -m atari_py.import_roms /home/hail/anaconda3/envs/suga_dreamer/lib/python3.9/site-packages/AutoROM/roms
```

###  Working Directory 📁
git clone 했을 때, 여러 개 폴더가 보이실 텐데 dreamerv3-torch 폴더를 제외한 나머지 폴더는 전부 SUGARL 에서 가져왔습니다.

저희가 main으로 해야할 게  "SUGARL에서 Sensory policy 부분을 DreamerV3로 교체하자" 입니다. 

main working directory는 suga_dreamerv3 폴더입니다.

일단 Atari에서 SAC를 사용했을 때, SUGARL + DreamerV3 가 잘 되는지를 보고자 하는 것이며, 추후에 DMC (2D, 3D로 확장할 예정입니다.)

<br>

### Current Progress (12/11)
1. action shape
처음에 random agent가 아무렇게나 했을 때, 저장된 episode 중 action 의 shape이 중구난방이다. motor 쪽은 환경의 action space에 따라 매번 달라지고 sensory는 x,y 위치로 정해져있다보니 이 둘을 따로 저장해야하지 않을까 하는 생각

<br>

### Must be fixed 🛠️ 

1. dreamer.py 디버깅 중에  지금은 config.task 를 "atari_pong" 으로 configs.yaml에서 고정시켜서 아마 이것만 할텐데 최종적으로는 atari에 대한 모든 환경을 config.task에 자동적으로 넣어서 돌아갈 수 있도록 만들어야 한다. 



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



### Notes

`scripts/` 안에 있는 scripts는 병렬 처리가 가능하도록, SURAGL 저자들이 shell script로 작성해둔 것입니다.

여기에서 Atari만 해도 환경을 26개를 넣어놨는데, 4090에서 3번 돌렸다가 전부 죽어버려서 `agent_dreamer/` 이 폴더에서 agent를 반복문으로 집어넣어주는 식으로 변경하였습니다. 다만 `scripts/` 안에 있는 몇몇 arguments 들이 필요할 수 있어 혹시나 남겨두었습니다.

main branch로 올라간 첫 버전은 그냥 SUGARL를 wandb에 찍을 수 있도록 몇개의 줄을 추가한 게 다이며, 추가 수정 사항 있을 시 README나 다른 수단으로 공유하도록 하겠습니다.



- 09/12 (Fri) Donggyu Lab meeting 

[PDF Download]({% raw %}_docs{% endraw %}/Active%20Vision%20Reinforcement%20Learning%20under%20Limited%20Vis...pdf)



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