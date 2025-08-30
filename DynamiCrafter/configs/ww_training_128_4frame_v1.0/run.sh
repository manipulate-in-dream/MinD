export HF_HOME="./.cache/huggingface"
export http_proxy=http://192.168.32.28:18000 
export https_proxy=http://192.168.32.28:18000
name="ww_training_128_4frame_v1.0"
config_file=configs/ww_training_128_4frame_v1.0/config.yaml
exp_name=${name}_oxe750k_robomind_franka_carrot_wrist
# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="./VGA/VgmACT/DynamiCrafter/save_checkpoints"

mkdir -p $save_root/${exp_name}

## run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=8 --nnodes=1 --master_addr=127.0.0.1 --master_port=1252 --node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name $exp_name \
--logdir $save_root \
--devices=8 \
lightning.trainer.num_nodes=1
# &>> ./$(date +%Y%m%d_%H%M%S).log &