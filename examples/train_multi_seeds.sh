for seed in $(seq 1 2)
do
    python train.py --algo happo_sr --env mamujoco --seed $seed --load_config ~/HARL/tuned_configs/mamujoco/Ant-v2-4x2/happo_sr/config.json
done