num_cards=8
bsz_cards=128
time=$(date +%Y%m%d_%H%M%S)
run_id="V3_DiTM_franka_oxe_all_11.7w+" # _${time}
mkdir ./${run_id}-image_aug

export HF_HOME=".cache/huggingface"
# export TFDS_DATA_DIR="open_x_embodiment/franka/merged_4tasks"
# export TFDS_DATA_DIR="open_x_embodiment/franka/whiteboard"
# export TFDS_DATA_DIR="open_x_embodiment/franka/pull_charger_wrist50"
export TFDS_DATA_DIR="/mnt/datasets/rtx_dataset_4"
# export TFDS_DATA_DIR="open_x_embodiment/franka/0720_rt1_rlbench_franka14"
# export TFDS_DATA_DIR="open_x_embodiment/franka/pick_place"
# export TFDS_DATA_DIR="open_x_embodiment/rlbench/dataset/rlbench_concat_future_fixed4"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=12345 --standalone --nnodes 1 --nproc-per-node $num_cards scripts/train_vgmvla.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom_finetuning \
  --vla.expected_world_size $num_cards \
  --vla.global_batch_size $(expr $bsz_cards \* $num_cards) \
  --vla.per_device_batch_size $bsz_cards \
  --vla.learning_rate 1e-5 \
  --run_root_dir "VGA/VgmACT" \
  --data_root_dir ${TFDS_DATA_DIR} \
  --image_aug True \
  --save_interval 1000 \
  --run_id ${run_id} \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --wandb_project "vgmact-oxefull-V3" \
  --wandb_entity 'litwellchi' \
  --is_resume False \
  --vgm_param_mode 'freeze' \
  --use_future_frame False \
  --use_full_ckpt_vgm False \
  --pretrained_checkpoint "/mnt//world_foundational_model/xiaowei/VGA/VgmACT/DynamiCrafter/save_checkpoints/ww_training_128_4frame_v1.0_oxe750k/checkpoints/trainstep_checkpoints/epoch=15-step=22500.ckpt" \
  --dataset_statistics_json "/mnt/datasets/rtx_dataset_4/custom_finetuning/1.0.0/dataset_statistics_4c18b30b6cd9b592e2418f46227966f6a88685bbcfcbba8d536b8c21fed9c23c.json" \
  --full_ckpt "VGA/VgmACT/V3_DiTM_franka_oxe_all--image_aug/checkpoints/step-017000-epoch-00-loss=0.0464.pt"
  # --pretrain_action_model 'CogACT/CogACT/CogACT-Base/checkpoints/CogACT-Base.pt' 
  # --pretrain_action_model 'videoact/CogACT/CogACT-Base/checkpoints/CogACT-Base.pt' 
  # --pretrain_action_model 'videoact/CogACT/CogACT-Base/checkpoints/CogACT-Base.pt' 
  # --pretrained_checkpoint "RoboCrafter/save_checkpoints/ww_training_128_4frame_v1.0_franka_4frame/checkpoints/trainstep_checkpoints/epoch=179-step=1800.ckpt" \
 


