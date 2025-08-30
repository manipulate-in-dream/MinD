import json
import csv

def process_jsonl(input_file, output_file):
    """
    读取 JSONL 文件，提取并转换数据，保存为 CSV 文件。
    
    Args:
        input_file (str): 输入的 JSONL 文件路径。
        output_file (str): 输出的 CSV 文件路径。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # 写入 CSV 表头
            csv_writer.writerow(['videoid', 'contentUrl', 'duration', 'page_dir', 'name'])
            
            for line in infile:
                # 解析 JSON 行
                data = json.loads(line.strip())
                
                # 提取并转换数据
                videoid = data['video_path'].split('/')[-1].replace('.mp4', '') if 'video_path' in data else ''
                contentUrl = videoid
                duration = 5  # 固定为 5
                page_dir = '/'.join(data['video_path'].split('/')[:-1]) if 'video_path' in data else ''
                name = data.get('lang', '')  # dense_lang 对应 name
                name = name.replace('"', '').replace('\t', '').replace('\n', '')  # 删除不需要的符号
                
                # 写入 CSV 行
                csv_writer.writerow([videoid, contentUrl, duration, page_dir, name])
        
        print(f"文件处理完成，已保存到 {output_file}")
    except Exception as e:
        print(f"处理文件时出错: {e}")

# 调用函数
input_file = '0627_agibot_droid_robomind_600k_longtext.jsonl'  # 输入文件名
output_file = 'DynamiCrafter/configs/ww_training_128_4frame_v1.0/0627_agibot_droid_robomind_600k.csv'  # 输出文件名
process_jsonl(input_file, output_file)