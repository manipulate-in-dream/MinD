#!/bin/bash

# MinD Training Script with Configurable Paths
# This script uses environment variables for all paths, making it portable across different systems

# Load environment variables from .env file if it exists
if [ -f "../.env" ]; then
    export $(cat ../.env | grep -v '^#' | xargs)
fi

# Training Configuration
num_cards=8
bsz_cards=128
time=$(date +%Y%m%d_%H%M%S)
run_id="${RUN_ID:-V3_DiTM_franka_oxe_all_11.7w+}" # Use environment variable or default

# Create output directory
output_dir="${MIND_OUTPUT_ROOT:-./outputs}/${run_id}-image_aug"
mkdir -p "${output_dir}"

# Set cache directories with defaults
export HF_HOME="${HF_HOME:-.cache/huggingface}"
export TFDS_DATA_DIR="${MIND_DATASET_ROOT:-/path/to/datasets}"

# Model paths with environment variable overrides
PRETRAINED_CHECKPOINT="${DYNAMICRAFTER_CHECKPOINT:-./checkpoints/dynamicrafter/epoch=15-step=22500.ckpt}"
DATASET_STATISTICS="${DATASET_STATISTICS_JSON:-./checkpoints/dataset_statistics.json}"
FULL_CHECKPOINT="${MIND_FULL_CHECKPOINT:-./checkpoints/step-017000-epoch-00-loss=0.0464.pt}"

# Validate required paths exist
echo "Checking required paths..."
for path_var in TFDS_DATA_DIR PRETRAINED_CHECKPOINT DATASET_STATISTICS; do
    path_value="${!path_var}"
    if [ ! -e "$path_value" ]; then
        echo "Warning: $path_var path does not exist: $path_value"
        echo "Please set the $path_var environment variable to a valid path"
    fi
done

# GPU configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi

# Parse the number of GPUs from CUDA_VISIBLE_DEVICES
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
actual_num_cards=${#GPU_ARRAY[@]}

if [ $actual_num_cards -ne $num_cards ]; then
    echo "Adjusting num_cards from $num_cards to $actual_num_cards based on CUDA_VISIBLE_DEVICES"
    num_cards=$actual_num_cards
fi

# Run training with torchrun
echo "Starting training with configuration:"
echo "  Run ID: ${run_id}"
echo "  Output Directory: ${output_dir}"
echo "  Number of GPUs: ${num_cards}"
echo "  Batch size per GPU: ${bsz_cards}"
echo "  Total batch size: $(expr $bsz_cards \* $num_cards)"
echo "  Dataset path: ${TFDS_DATA_DIR}"
echo "  Pretrained checkpoint: ${PRETRAINED_CHECKPOINT}"
echo ""

torchrun \
  --master_port=${MASTER_PORT:-12345} \
  --standalone \
  --nnodes 1 \
  --nproc-per-node $num_cards \
  scripts/train_vgmvla.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix custom_finetuning \
  --vla.expected_world_size $num_cards \
  --vla.global_batch_size $(expr $bsz_cards \* $num_cards) \
  --vla.per_device_batch_size $bsz_cards \
  --vla.learning_rate ${LEARNING_RATE:-1e-5} \
  --run_root_dir "${RUN_ROOT_DIR:-VGA/VgmACT}" \
  --data_root_dir "${TFDS_DATA_DIR}" \
  --image_aug ${IMAGE_AUG:-True} \
  --save_interval ${SAVE_INTERVAL:-1000} \
  --run_id "${run_id}" \
  --repeated_diffusion_steps ${DIFFUSION_STEPS:-8} \
  --future_action_window_size ${ACTION_WINDOW:-15} \
  --action_model_type ${ACTION_MODEL_TYPE:-DiT-B} \
  --wandb_project "${WANDB_PROJECT:-vgmact-oxefull-V3}" \
  --wandb_entity "${WANDB_ENTITY:-litwellchi}" \
  --is_resume ${IS_RESUME:-False} \
  --vgm_param_mode "${VGM_PARAM_MODE:-freeze}" \
  --use_future_frame ${USE_FUTURE_FRAME:-False} \
  --use_full_ckpt_vgm ${USE_FULL_CKPT_VGM:-False} \
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --dataset_statistics_json "${DATASET_STATISTICS}" \
  --full_ckpt "${FULL_CHECKPOINT}"

# Optional: Save the configuration used for this run
echo "Saving configuration to ${output_dir}/config.txt"
cat > "${output_dir}/config.txt" << EOF
Run ID: ${run_id}
Timestamp: ${time}
Number of GPUs: ${num_cards}
Batch size per GPU: ${bsz_cards}
Total batch size: $(expr $bsz_cards \* $num_cards)
Dataset path: ${TFDS_DATA_DIR}
Pretrained checkpoint: ${PRETRAINED_CHECKPOINT}
Dataset statistics: ${DATASET_STATISTICS}
Full checkpoint: ${FULL_CHECKPOINT}
Learning rate: ${LEARNING_RATE:-1e-5}
Diffusion steps: ${DIFFUSION_STEPS:-8}
Action window: ${ACTION_WINDOW:-15}
Action model type: ${ACTION_MODEL_TYPE:-DiT-B}
EOF

echo "Training completed. Output saved to ${output_dir}"