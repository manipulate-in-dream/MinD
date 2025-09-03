import argparse
import os
from pathlib import Path
import numpy as np
import sys

import hydra
import torch
from pytorch_lightning import seed_everything

# Add parent directory to path to import config module
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_path_config

from policy_evaluation.utils import get_default_beso_and_env
from policy_models.utils.utils import get_last_checkpoint


def model_load(args):
    """
    根据参数加载和配置模型、环境和语言嵌入。

    Args:
        args (argparse.Namespace): 包含所有必要路径和配置的参数，例如：
                                     - action_model_folder (str): 动作模型检查点所在的文件夹。
                                     - video_model_path (str): 预训练视频模型的路径。
                                     - clip_model_path (str): 预训练CLIP模型的路径。
                                     - calvin_abc_dir (str): CALVIN 数据集的根目录。
                                     - device (int): 使用的 GPU ID。
                                     - num_sampling_steps (int): 采样步数。
                                     - ... (其他在配置文件中定义的参数)

    Returns:
        dict: 包含加载好的 'model', 'env', 'lang_embeddings' 和 'cfg' 的字典。
    """
    # 1. 使用 Hydra 初始化配置
    # Load default config and overwrite with command line arguments
    with hydra.initialize(config_path="./policy_conf", job_name="calvin_evaluate_all"):
        cfg = hydra.compose(config_name="calvin_evaluate_all.yaml")

    # 2. 使用 args 中的参数覆盖默认配置
    cfg.train_folder = args.action_model_folder
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.device = args.device
    if hasattr(args, 'num_sampling_steps'):
        cfg.num_sampling_steps = args.num_sampling_steps
    # 可以根据需要添加更多参数覆盖
    
    # 3. 设置随机种子和设备
    seed_everything(0, workers=True)
    device = torch.device(f"cuda:{cfg.device}")
    torch.cuda.set_device(device)

    # 4. 获取环境和语言嵌入
    # This function sets up the environment and language embeddings based on the config.
    # The checkpoint is used to find the correct configuration files.
    checkpoint_for_env = get_last_checkpoint(Path(cfg.train_folder))
    
    # 5. 实例化并加载模型权重
    # Find the specific model checkpoint to load
    model_ckpt_path = get_last_checkpoint(Path(cfg.train_folder))
    if not model_ckpt_path:
        raise FileNotFoundError(f"No checkpoint file found in {cfg.train_folder}")
        
    print(f"Loading model from {model_ckpt_path}")
    state_dict = torch.load(model_ckpt_path, map_location='cpu')
    
    # Instantiate model from config
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(state_dict['model'], strict=False)
    
    # 6. 配置模型推理参数
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler

    # 7. 将模型移至指定设备并设置为评估模式
    model.freeze()
    model = model.to(device)
    model.process_device()
    model.eval()
    
    print("Model loaded successfully.")

    return model

def model_predict(loaded_assets, obs, prompt):
    """
    使用加载好的模型，根据当前观测和语言指令预测动作。

    Args:
        loaded_assets (dict): 从 model_load 函数返回的字典。
        obs (dict): 从环境中获取的当前观测。
        prompt (str): 语言指令，例如 "pick up the blue block"。

    Returns:
        torch.Tensor: 模型预测的动作。
    """
    # 1. 从 loaded_assets 中解包所需对象
    model = loaded_assets["model"]
    goal = dict()
    goal["lang_text"] = prompt

    # 3. 执行模型单步预测
    # The `step` method corresponds to predicting one action
    action = model.step(obs, goal)
    
    return action


if __name__ == '__main__':
    # ============================ 示例用法 ============================
    # 使用 argparse 来解析命令行参数，与原始脚本保持一致
    parser = argparse.ArgumentParser()
    # Load configuration
    config = get_path_config()
    
    # Use environment variables or config-based defaults
    default_action_model = os.getenv('ACTION_MODEL_FOLDER', 
                                     str(config.get_model_path('action', 'dp-calvin')))
    default_video_model = os.getenv('VIDEO_MODEL_PATH',
                                    str(config.get_model_path('video', 'vpp_rlbench')))
    default_input_image = os.getenv('INPUT_IMAGE_PATH',
                                    str(config.data_root / 'test_images' / 'test.png'))
    
    parser.add_argument("--action_model_folder", type=str, default=default_action_model)
    parser.add_argument("--video_model_path", type=str, default=default_video_model)
    parser.add_argument("--clip_model_path", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_sampling_steps", type=int, default=10)
    
    parser.add_argument("--input_image_path", type=str, default=default_input_image, 
                       help="机器人当前观测的图像路径")
    parser.add_argument("--instruction", type=str, default="pick up the red block", help="需要执行的文本指令")
    args = parser.parse_args()
    
    lang = ""
    from PIL import Image
    import cv2
    obs = cv2.cvtColor(cv2.imread(args.input_image_path), cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(obs, (256, 256), interpolation=cv2.INTER_AREA)
    observation_image = np.array(resized_image)
    # 验证路径是否存在
    paths_to_check = [
        (args.action_model_folder, "action_model_folder"),
        (args.video_model_path, "video_model_path"),
        (args.input_image_path, "input_image_path")
    ]
    
    all_paths_valid = True
    for path, name in paths_to_check:
        if not os.path.exists(path):
            print(f"Warning: {name} not found at '{path}'")
            print(f"Please set the {name.upper()} environment variable or update config.yaml")
            all_paths_valid = False
    
    if not all_paths_valid:
        print("\nSome paths are missing. The model may not load correctly.")
        print("Continuing anyway...")
    if all_paths_valid:
        # 1. 加载模型
        loaded_assets = model_load(args)
        obs = dict()
        obs["rgb_obs"] = {}
        float_image = observation_image.astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(float_image)
        final_tensor = tensor_image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        obs["rgb_obs"]['rgb_static'] = final_tensor
        model = loaded_assets["model"]
        model.use_text_not_embedding = True
        model.reset() # 重置模型内部状态（例如RNN）

        # 3. 定义一个任务并进行预测
        task_prompt = "lift the red block"
        print(f"\nExecuting prediction for task: '{task_prompt}'")

        predicted_action = model_predict(loaded_assets, obs, task_prompt)

        print(f"Predicted Action Shape: {predicted_action.shape}")
        print(f"Predicted Action (first 5 dims): {predicted_action[:5].cpu().numpy()}")

        