***
## Installation:
Modify prefix in freeze.yml according to your anaconda environment path.
```bash
 conda env create -f freeze.yml
 conda activate pretrainmol3d
 ```

 ## Training
 * For 8 regression tasks: "matbench_jdft2d","matbench_dielectric", "matbench_phonons","matbench_perovskites","matbench_log_gvrh",
 "matbench_log_kvrh","matbench_mp_e_form","matbench_mp_gap",
 ```bash
python train_matbench.py --dataset_name=matbench_jdft2d --normalize --node-attn --use-bn
 ```
 * For 1 classification task: "matbench_mp_is_metal"
  ```bash
python train_matbench.py --dataset_name=matbench_mp_is_metal --node-attn --use-bn
 ```
 <!-- ## Finetuning
```shell
bash run_finetune.sh --num-layers 12 --batch-size 128 \
        --dropout 0.3 --dataset ogbg-molpcba \
        --pooler-dropout 0.1 --epochs 50 --seed 42 \
        -m /yourpretrainedmodel \
        --lr 0.0005 --weight-decay 0.01 --grad-norm 1 --prefix molpcba
``` -->