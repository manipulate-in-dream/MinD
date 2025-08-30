num_cards=4
bsz_cards=32
time=$(date +%Y%m%d_%H%M%S)
run_id="V3_DiTM_franka_gkz_all" # _${time}
mkdir ./${run_id}-image_aug

export HF_HOME=".cache/huggingface"

export TFDS_DATA_DIR="open_x_embodiment/franka/gkz_rlds"


CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --standalone --nnodes 1 --nproc-per-node $num_cards scripts/train_vgmvla.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom_finetuning \
  --vla.expected_world_size $num_cards \
  --vla.global_batch_size $(expr $bsz_cards \* $num_cards) \
  --vla.per_device_batch_size $bsz_cards \
  --vla.learning_rate 1e-5 \
  --run_root_dir "videoact/VgmACT" \
  --data_root_dir ${TFDS_DATA_DIR} \
  --image_aug True \
  --save_interval 1000 \
  --run_id ${run_id} \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --wandb_project "vgmact-franka4-V3" \
  --wandb_entity 'litwellchi' \
  --is_resume False \
  --vgm_param_mode 'freeze' \
  --use_future_frame False \
  --use_full_ckpt_vgm False \
  --pretrained_checkpoint "RoboCrafter/save_checkpoints/ww_training_128_4frame_v1.0_pdfranka_4frame_600k/checkpoints/trainstep_checkpoints/epoch=37-step=600.ckpt" \
  --dataset_statistics_json "open_x_embodiment/franka/gkz_rlds/data_statistics.json" \
  --full_ckpt "videoact/VgmACT/V3_DiTM_franka_pd_data_pretrain_wrist_1--image_aug/checkpoints/step-010000-epoch-07-loss=0.0599.pt"

