import json
import os
import shutil
from pathlib import Path

def create_sample_dataset(metadata_file, target_dir, num_samples=10):
    """
    从metadata_bbox_ref.json按照多级排序筛选数据，复制相关文件到目标目录
    
    筛选逻辑：
    1. aesthetic score 排前 3n 条
    2. 在前面结果中，motion score 排前 2n 条  
    3. 在前面结果中，temporal consistency score 排前 n 条
    
    Args:
        metadata_file: 源metadata_bbox_ref.json文件路径
        target_dir: 目标目录路径
        num_samples: 最终要获取的样本数量，默认10
    """
    print(f"正在读取 {metadata_file}...")
    
    # 读取原始metadata文件
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"原始数据包含 {len(metadata)} 个条目")
    
    # 第一步：按aesthetic score排序，取前3n条
    print(f"第一步：按aesthetic score排序，取前 {3 * num_samples} 条...")
    metadata_aesthetic = sorted(metadata, key=lambda x: x.get('aesthetic_score', 0), reverse=True)
    top_aesthetic = metadata_aesthetic[:3 * num_samples]
    print(f"aesthetic score筛选后剩余 {len(top_aesthetic)} 条")
    
    # 第二步：在前面结果中按motion score排序，取前2n条
    print(f"第二步：按motion score排序，取前 {2 * num_samples} 条...")
    top_aesthetic_motion = sorted(top_aesthetic, key=lambda x: x.get('motion_score', 0), reverse=True)
    top_motion = top_aesthetic_motion[:2 * num_samples]
    print(f"motion score筛选后剩余 {len(top_motion)} 条")
    
    # 第三步：在前面结果中按temporal consistency score排序，取前n条
    print(f"第三步：按temporal consistency score排序，取前 {num_samples} 条...")
    top_motion_temporal = sorted(top_motion, key=lambda x: x.get('temporal_consistency_score', 0), reverse=True)
    final_sample_data = top_motion_temporal[:num_samples]
    print(f"temporal consistency score筛选后最终获得 {len(final_sample_data)} 条")
    
    # 转换为训练格式
    train_data = []
    skipped_count = 0
    
    for item in final_sample_data:
        # 检查fg_video_path是否为空，如果为空则跳过
        if not item.get("fg_video_path") or item["fg_video_path"].strip() == "":
            skipped_count += 1
            continue
            
        # 创建新的训练数据条目
        train_item = {
            "file_path": item["video_path"],
            "text": item["caption"],
            "type": "video",
            "ref": item["fg_video_path"]
        }
        
        train_data.append(train_item)
    
    if skipped_count > 0:
        print(f"跳过了 {skipped_count} 个ref为空的样本")
    
    print(f"最终将复制 {len(train_data)} 条有效数据")
    
    # 确保目标目录存在
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"创建目标目录: {target_dir}")
    
    # 获取源数据目录
    source_dir = Path(metadata_file).parent
    
    # 复制文件并更新路径
    copied_files = 0
    failed_files = 0
    
    for i, item in enumerate(train_data):
        print(f"处理第 {i+1}/{len(train_data)} 条数据...")
        
        # 复制主视频文件
        source_video_path = source_dir / item["file_path"]
        target_video_path = target_path / item["file_path"]
        
        # 确保目标视频目录存在
        target_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if source_video_path.exists():
                shutil.copy2(source_video_path, target_video_path)
                copied_files += 1
                print(f"  ✓ 复制主视频: {item['file_path']}")
            else:
                print(f"  ✗ 主视频文件不存在: {source_video_path}")
                failed_files += 1
        except Exception as e:
            print(f"  ✗ 复制主视频失败: {e}")
            failed_files += 1
        
        # 复制参考视频文件
        source_ref_path = source_dir / item["ref"]
        target_ref_path = target_path / item["ref"]
        
        # 确保目标参考视频目录存在
        target_ref_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if source_ref_path.exists():
                shutil.copy2(source_ref_path, target_ref_path)
                copied_files += 1
                print(f"  ✓ 复制参考视频: {item['ref']}")
            else:
                print(f"  ✗ 参考视频文件不存在: {source_ref_path}")
                failed_files += 1
        except Exception as e:
            print(f"  ✗ 复制参考视频失败: {e}")
            failed_files += 1
    
    # 保存新的train.json
    target_train_file = target_path / "train.json"
    with open(target_train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n复制完成！")
    print(f"成功复制 {copied_files} 个文件")
    print(f"失败 {failed_files} 个文件")
    print(f"已保存 {len(train_data)} 个条目到 {target_train_file}")
    
    # 显示一个示例
    if train_data:
        print("\n示例数据:")
        print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

def main():
    # 文件路径配置
    metadata_file = "datasets/synworld11k/metadata_bbox_ref.json"
    num_samples = 12
    target_dir = f"datasets/synworld{num_samples}"
    
    # 检查源文件是否存在
    if not os.path.exists(metadata_file):
        print(f"错误: 源文件 {metadata_file} 不存在")
        return
    
    # 执行复制
    try:
        create_sample_dataset(metadata_file, target_dir, num_samples)
        print("脚本执行完成！")
    except Exception as e:
        print(f"执行过程中出现错误: {e}")

if __name__ == "__main__":
    main() 