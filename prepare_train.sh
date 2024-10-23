#!/bin/bash

docker compose up -d

# 学習結果などを削除(表面的に同条件のものがあれば、すぐ終わるので))
sudo rm -rf frame_field_learning/runs/mapping_dataset.unet_resnet101_pretrained\ \|\ 2024-10-17\ 03\:09\:35//*
# 余計なセッションを削除(邪魔なので)
screen -ls | grep Detached | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit

sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# ログに保存
df -hT > prepare.log
echo >> prepare.log
free -h >> prepare.log

# 学習開始
screen -dmS train ffl -c '
    docker compose exec ffl bash -c "
        python src/main.py fit  -c config/trainer.yml -c config/data_mitaka_kawasaki_on_memory_DSM_grad.yml \
                                -c config/optim_adamw.yml -c config/model_unet-test.yml"  1> output/stdout.log 2> output/stderr.log
'