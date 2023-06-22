for seed in $(seq 1 3)
do
    python train.py --algo happo --env mamujoco --seed $seed --load_config ~/HARL/tuned_configs/mamujoco/Humanoid-v2-17x1/happo/config.json
done


for seed in $(seq 1 3)
do
    python train.py --algo happo_sr --env mamujoco --seed $seed --load_config ~/HARL/tuned_configs/mamujoco/Humanoid-v2-17x1/happo_sr/config.json
done