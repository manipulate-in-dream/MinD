import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow_datasets as tfds
import cv2
import multiprocessing as mp


def save_images_as_video(images, output_file, fps=15):
    height, width, _ = np.array(images[0]).shape
    size = (width, height)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for image in images:
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        out.write(img_array)
    out.release()


def interpolate_frames(images, target_frame_count=4):
    current_len = len(images)
    if current_len >= target_frame_count:
        return images[:target_frame_count]

    new_images = []
    for i in range(target_frame_count):
        pos = i * (current_len - 1) / (target_frame_count - 1)
        low, high = int(np.floor(pos)), int(np.ceil(pos))
        alpha = pos - low
        if low == high:
            new_images.append(images[low])
        else:
            blended = Image.blend(images[low], images[high], alpha)
            new_images.append(blended)
    return new_images


def process_range(args):
    dataset_name, display_key, root_dir, start_idx, end_idx, position = args
    print(f"[Worker {position}] PID {os.getpid()} | Range=({start_idx}-{end_idx}) | Starting...")

    dataset_path = os.path.join(root_dir, dataset_name, '1.0.0')
    builder = tfds.builder_from_directory(builder_dir=dataset_path)
    dataset = builder.as_dataset(split='train')

    video_dir = os.path.join(root_dir, 'videos', dataset_name)
    os.makedirs(video_dir, exist_ok=True)

    metadata = []
    total = end_idx - start_idx

    with tqdm(
        total=total,
        position=position,
        desc=f"Worker {position} [{start_idx}-{end_idx})",
        leave=True,
        dynamic_ncols=True,
        miniters=1,
        mininterval=0.5
    ) as pbar:
        for idx, episode in enumerate(dataset):
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break

            try:
                steps = episode['steps']
                images = [Image.fromarray(step['observation'][display_key].numpy()) for step in steps]
                images = interpolate_frames(images, target_frame_count=4)

                instructions = [step['language_instruction'].numpy().decode('utf-8') for step in steps]
                instruction = instructions[0] if instructions else "No instruction"

                video_path = os.path.join(video_dir, f"{idx}.mp4")
                save_images_as_video(images, video_path)

                metadata.append({
                    'videoid': idx,
                    'contentUrl': dataset_name,
                    'duration': len(instructions) / 15,
                    'page_dir': video_dir,
                    'name': instruction,
                })

            except Exception as e:
                print(f"[Worker {position}] Error on idx {idx}: {e}")

            pbar.update(1)

    return metadata


def process_dataset_multiprocess(dataset_name, display_key='image',
                                 root_dir='/mnt/datasets/rtx_dataset_4',
                                 max_videos=None, num_workers=4):
    dataset_path = os.path.join(root_dir, dataset_name, '1.0.0')
    builder = tfds.builder_from_directory(builder_dir=dataset_path)

    total_available = builder.info.splits['train'].num_examples
    total = min(total_available, max_videos) if max_videos else total_available
    print(f"\n🔢 Total episodes to process: {total}")

    chunk_size = (total + num_workers - 1) // num_workers
    ranges = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        ranges.append((dataset_name, display_key, root_dir, start, end, i))

    print("\n🧩 Worker Ranges:")
    for r in ranges:
        print(f"  Worker {r[5]}: [{r[3]} - {r[4]})")

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool:
        results = pool.map(process_range, ranges)

    metadata = []
    for result in results:
        metadata.extend(result)

    return metadata


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    datasets_to_process = ['rtx_dataset']
    total_metadata = []

    for dataset in datasets_to_process:
        metadata = process_dataset_multiprocess(
            dataset_name=dataset,
            display_key='image',
            max_videos=None,     # None 表示处理全部数据
            num_workers=4         # 可根据 CPU 核数调整
        )
        total_metadata.extend(metadata)

    # 保存元数据到 CSV
    metadata_df = pd.DataFrame(total_metadata)
    save_path = '/mnt/datasets/rtx_dataset_4/videos_oxemind.csv'
    metadata_df.to_csv(save_path, mode='a', index=False)
    print(f"\n✅ Metadata saved to: {save_path}")