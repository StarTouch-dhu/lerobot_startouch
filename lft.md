# 虚拟环境
conda create -y -n lerobot_v31 python=3.10
conda activate lerobot_v31
pip install -e .




# 数据采集
python src/lerobot/scripts/lerobot_record.py \
    --robot.type=startouch_arm_single \
    --dataset.repo_id=left/startouch \
    --dataset.root=/home/lft/workspace/python_project/VLA/data_v31/startouch_test2 \
    --dataset.single_task="test" \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --display_data=true





# 训练
安装依赖
conda install -c conda-forge ffmpeg



python src/lerobot/scripts/lerobot_train.py  \
  --dataset.repo_id=left/startouch \
  --dataset.root=/home/lft/workspace/python_project/VLA/data_v31/startouch_test3 \
  --policy.type=act \
  --output_dir=outputs/ACT/train/act_3_2 \
  --job_name=act_startouch \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false


# 推理

python src/lerobot/scripts/inference_startouch_dual.py
