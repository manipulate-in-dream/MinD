import socket
import pickle
import numpy as np
import zlib

from PIL import Image
from vla import load_vgmvla
import torch


def model_load():
     model = load_vgmvla(
         "./Dynamicrafter/checkpoints/epoch=37-step=600.ckpt",
         load_for_training=False,
         action_model_type='DiT-B',
         future_action_window_size=15,
         action_dim=7,
         vgm_param_mode="freeze",
         full_ckpt="./checkpoints/step-005000-epoch-102-loss=0.0262.pt",
         hf_token="",
         dataset_statistics_json="./checkpoints/dataset_statistics.json",
         video_project_model_mb=100,
         is_eval=True
         )
     model.to(f'cuda:{0}').eval()
     return model
 
def model_predict(model, image, prompt, ddim_step=10, save_video_path=None, pred_dynamic=True, use_dual=True):
     actions, _, _  = model.predict_action(
               image,
               prompt,
               unnorm_key='custom_finetuning',
               cfg_scale = 1.5, 
               use_ddim = True,
               num_ddim_steps = ddim_step,
               video_save_path = save_video_path,
               pred_dynamic = pred_dynamic,
               return_pred_dynamic = use_dual
               )
     return actions[0]
 
def compressed_numpy_to_image(img_bytes):
    decompressed = zlib.decompress(img_bytes)
    np_img = np.frombuffer(decompressed, dtype=np.uint8).reshape(224, 224, 3)
    return Image.fromarray(np_img)

def recv_all(conn, buffer_size=4096):
    data = b""
    while True:
        part = conn.recv(buffer_size)
        data += part
        if len(part) < buffer_size:
            break
    return data

model = model_load()

save_video_path = "./predicted_videos/gkz_pick_place_5k"
# 2. 创建Socket
HOST = '0.0.0.0'  # 监听所有IP
PORT = 12138      # 选择一个端口（确保防火墙允许）
count = 0
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Server started, waiting for connections...")
    while True:
        conn, addr = s.accept()
        print(f"Connection accepted from {addr}")
        with conn:
            print("[Server] 客户端已连接，准备接收数据...")
            length_bytes = conn.recv(4)
            if not length_bytes:
                print("[Server] 错误：客户端未发送数据长度")
                raise ValueError("Client disconnected before sending data")
            
            length = int.from_bytes(length_bytes, byteorder='big')
            print(f"[Server] 客户端数据长度: {length} bytes")
            
            received_data = b''
            while len(received_data) < length:
                chunk = conn.recv(min(4096, length - len(received_data)))
                if not chunk:
                    print("[Server] 错误：客户端数据不完整")
                    raise ConnectionError("Incomplete data received")
                received_data += chunk
            
            print("[Server] 数据接收完成，开始反序列化...")
            try:
                img_bytes, instruction, exp_num, timestep = pickle.loads(received_data)
                print("[Server] 数据反序列化成功")
            except Exception as e:
                print("[Server] 反序列化失败:", e)
                raise

            print("[Server] 调用 model_predict...")
            try:
                video_path = save_video_path + f"/{instruction}_{exp_num}/timestep_{str(timestep)}"
                action = model_predict(model, compressed_numpy_to_image(img_bytes), instruction, save_video_path=video_path)
                print("[Server] action:", action)  # 确认打印
            except Exception as e:
                print("[Server] model_predict 报错:", e)
                raise

            action_data = pickle.dumps(action)
            print(f"[Server] 准备返回 action (长度: {len(action_data)} bytes)")
            conn.sendall(len(action_data).to_bytes(4, byteorder='big'))
            conn.sendall(action_data)
            print("[Server] 数据已发送回客户端")
            count += 1