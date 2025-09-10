import json
import os

def convert_metadata_to_train_format(input_file, output_file):
    """
    将metadata_bbox_ref.json转换为训练格式的train.json
    
    Args:
        input_file: 输入的metadata_bbox_ref.json文件路径
        output_file: 输出的train.json文件路径
    """
    print(f"正在读取 {input_file}...")
    
    # 读取原始metadata文件
    with open(input_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"原始数据包含 {len(metadata)} 个条目")
    
    # 转换数据格式
    train_data = []
    
    for item in metadata:
        # 创建新的训练数据条目
        train_item = {
            "file_path": item["video_path"],
            "text": item["caption"],
            "type": "video",
            "ref": item["fg_video_path"]
        }
        
        train_data.append(train_item)
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！已保存 {len(train_data)} 个条目到 {output_file}")
    
    # 显示一个示例
    if train_data:
        print("\n示例数据:")
        print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

def main():
    # 文件路径
    input_file = "datasets/synworld_11K/metadata_bbox_ref.json"
    output_file = "datasets/synworld_11K/train.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 执行转换
    try:
        convert_metadata_to_train_format(input_file, output_file)
        print("脚本执行完成！")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

if __name__ == "__main__":
    main()