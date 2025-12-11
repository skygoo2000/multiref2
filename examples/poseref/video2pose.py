#!/usr/bin/env python3
"""
视频姿态提取脚本
从视频中提取相机姿态信息，生成pose JSON文件用于姿态引导的视频生成

功能:
    - 使用SAM3分割视频中的物体
    - 将背景替换为白色
    - 将物体居中裁剪
    - 提取相机姿态轨迹（旋转和平移）
    - 生成pose JSON文件

用法:
    python video2pose.py --video_path <视频路径> --text_prompt <物体描述> --output_dir <输出目录>

示例:
    python video2pose.py --video_path input.mp4 --text_prompt "a car" --output_dir ./samples/sam3/car
    python video2pose.py --video_path input.mp4 --text_prompt "person" --device cuda:0
    python video2pose.py --video_path input.mp4 --text_prompt "cat" --device cuda:1 --padding 20
    python video2pose.py --video_path input.mp4 --text_prompt "drone" --max_frames 49

输出文件:
    - centered_video.mp4: 居中裁剪后的视频（用作mask）
    - white_background_video.mp4: 白色背景的视频（用作ref）
    - pose.json: 相机姿态轨迹文件
    - metadata.json: 包含prompt、ref、mask、pose路径的元数据
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video
import cv2
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation

try:
    from moviepy.editor import ImageSequenceClip
except:
    from moviepy import ImageSequenceClip

def parse_args():
    parser = argparse.ArgumentParser(description="使用SAM3处理视频，分割物体并替换背景")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="输入视频路径"
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help="要分割的物体描述，例如: 'person', 'cat', 'car'"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples/sam3",
        help="输出目录路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/sam3",
        help="SAM3模型名称"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="最大处理帧数，None表示处理所有帧"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="输出视频帧率，None表示使用原视频帧率"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="裁剪时在物体周围添加的padding像素数"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="使用的设备，例如: 'cuda:0', 'cuda:1', 'cpu'"
    )
    parser.add_argument(
        "--enable_crop",
        action="store_true",
        help="启用追踪裁剪（默认关闭）"
    )
    parser.add_argument(
        "--process_fps",
        type=int,
        default=16,
        help="处理视频时的采样帧率（默认16fps）"
    )
    parser.add_argument(
        "--extract_camera",
        action="store_true",
        help="提取相机外参（使用Depth Anything 3）"
    )
    parser.add_argument(
        "--da3_model",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE",
        help="Depth Anything 3模型名称"
    )
    return parser.parse_args()


def load_sam3_model(model_name, device):
    """加载SAM3模型和处理器"""
    print(f"正在加载SAM3模型: {model_name}")
    model = Sam3VideoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained(model_name)
    print("模型加载完成")
    return model, processor


def resize_frames(frames, max_short_side=480):
    """调整视频帧大小，确保短边不超过max_short_side"""
    h, w = frames[0].shape[:2]
    short_side = min(h, w)
    
    if short_side <= max_short_side:
        return frames
    
    # 计算缩放比例
    scale = max_short_side / short_side
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    print(f"调整分辨率: {w}x{h} -> {new_w}x{new_h}")
    
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_frames.append(resized)
    
    return np.array(resized_frames)


def load_images_from_folder(folder_path):
    """从文件夹加载所有图片并拼合成视频"""
    print(f"正在从文件夹加载图片: {folder_path}")
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # 获取所有图片文件
    image_files = []
    for file in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        raise ValueError(f"文件夹中没有找到图片文件: {folder_path}")
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 加载所有图片
    frames = []
    for img_path in tqdm(image_files, desc="加载图片"):
        img = cv2.imread(img_path)
        if img is not None:
            # OpenCV读取的是BGR格式，转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    
    if not frames:
        raise ValueError(f"无法加载任何图片: {folder_path}")
    
    video_frames = np.array(frames)
    fps = 1.0  # 图片序列默认1fps
    
    print(f"加载完成: {len(video_frames)} 张图片, FPS: {fps}")
    return video_frames, fps


def load_video_frames(video_path, max_short_side=480, max_fps=16, process_fps=16):
    """加载视频帧或图片序列，并限制分辨率和帧率"""
    
    # 检查是否是文件夹（图片序列）
    if os.path.isdir(video_path):
        video_frames, fps = load_images_from_folder(video_path)
        # 图片序列不进行帧率采样，处理所有帧
        print(f"图片序列模式: 处理所有 {len(video_frames)} 帧")
    elif video_path.startswith("http://") or video_path.startswith("https://"):
        # 网络视频
        print(f"正在加载网络视频: {video_path}")
        video_frames, fps = load_video(video_path)
    else:
        # 本地视频文件
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        print(f"正在加载视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV读取的是BGR格式，转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        video_frames = np.array(frames)
        print(f"原始视频: {len(video_frames)} 帧, FPS: {fps}")
    
    # 只对视频文件进行帧率采样（图片序列不采样）
    if not os.path.isdir(video_path):
        original_fps = fps
        original_frame_count = len(video_frames)
        
        # 使用process_fps进行采样
        if fps > process_fps:
            # 计算采样间隔（向上取整以确保采样）
            frame_interval = max(1, round(fps / process_fps))
            
            # 如果间隔大于1，进行采样
            if frame_interval > 1:
                video_frames = video_frames[::frame_interval]
                fps = original_fps / frame_interval
                print(f"采样处理: 每{frame_interval}帧采样1帧 ({original_frame_count}帧 -> {len(video_frames)}帧, {original_fps:.2f}fps -> {fps:.2f}fps)")
        
        # 限制输出FPS（如果采样后仍然超过max_fps）
        if fps > max_fps:
            additional_interval = max(1, round(fps / max_fps))
            if additional_interval > 1:
                video_frames = video_frames[::additional_interval]
                fps = fps / additional_interval
                print(f"限制输出FPS: {len(video_frames)}帧, {fps:.2f}fps")
    
    # 限制分辨率
    video_frames = resize_frames(video_frames, max_short_side)
    
    print(f"处理后视频: {len(video_frames)} 帧, FPS: {fps:.2f}")
    return video_frames, fps


def process_video_with_sam3(model, processor, video_frames, text_prompt, device, max_frames=None):
    """使用SAM3处理视频，获取每一帧的bbox和mask，只保留前50%置信度的结果"""
    print(f"正在使用SAM3处理视频，物体提示: '{text_prompt}'")
    
    # 如果指定了最大帧数，则截取
    if max_frames is not None and len(video_frames) > max_frames:
        video_frames = video_frames[:max_frames]
        print(f"限制处理帧数为: {max_frames}")
    
    # 初始化视频推理会话
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )
    
    # 添加文本提示
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=text_prompt,
    )
    
    # 处理所有帧
    outputs_per_frame = {}
    max_frame_num = len(video_frames) if max_frames is None else min(max_frames, len(video_frames))
    
    print("正在处理视频帧...")
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session,
        max_frame_num_to_track=max_frame_num
    ):
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        
        # 只保留前50%置信度的结果
        if len(processed_outputs['object_ids']) > 0:
            scores = processed_outputs['scores']
            
            # 计算置信度阈值（前50%）
            if len(scores) > 1:
                sorted_scores = torch.sort(scores, descending=True)[0]
                threshold_idx = len(sorted_scores) // 2
                threshold = sorted_scores[threshold_idx].item()
                
                # 过滤低置信度的结果
                keep_mask = scores >= threshold
                
                filtered_outputs = {
                    'object_ids': processed_outputs['object_ids'][keep_mask],
                    'scores': processed_outputs['scores'][keep_mask],
                    'boxes': processed_outputs['boxes'][keep_mask],
                    'masks': processed_outputs['masks'][keep_mask],
                }
                outputs_per_frame[model_outputs.frame_idx] = filtered_outputs
            else:
                outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        else:
            outputs_per_frame[model_outputs.frame_idx] = processed_outputs
    
    print(f"处理完成: {len(outputs_per_frame)} 帧")
    return outputs_per_frame


def get_bounding_box_from_outputs(outputs):
    """从SAM3输出中获取边界框（使用boxes而不是mask）"""
    # outputs包含boxes字段，格式为XYXY (absolute coordinates)
    if len(outputs['object_ids']) == 0:
        return None
    
    # 使用第一个检测到的物体（通常是最主要的物体）
    box = outputs['boxes'][0].cpu().numpy()  # shape: (4,) [x_min, y_min, x_max, y_max]
    
    x_min, y_min, x_max, y_max = box
    return int(x_min), int(y_min), int(x_max), int(y_max)


def smooth_trajectory(centers, window_size=5):
    """使用移动平均平滑物体中心轨迹"""
    if len(centers) == 0:
        return centers
    
    # 转换为numpy数组
    indices = sorted(centers.keys())
    x_coords = np.array([centers[i][0] for i in indices])
    y_coords = np.array([centers[i][1] for i in indices])
    
    # 使用卷积进行移动平均
    kernel = np.ones(window_size) / window_size
    
    # 处理边界：使用valid模式，然后填充边界
    if len(x_coords) > window_size:
        x_smooth = np.convolve(x_coords, kernel, mode='same')
        y_smooth = np.convolve(y_coords, kernel, mode='same')
        
        # 修正边界效应
        for i in range(window_size // 2):
            x_smooth[i] = np.mean(x_coords[:i + window_size // 2 + 1])
            y_smooth[i] = np.mean(y_coords[:i + window_size // 2 + 1])
            x_smooth[-(i+1)] = np.mean(x_coords[-(i + window_size // 2 + 1):])
            y_smooth[-(i+1)] = np.mean(y_coords[-(i + window_size // 2 + 1):])
    else:
        x_smooth = x_coords
        y_smooth = y_coords
    
    # 转换回字典
    smoothed_centers = {indices[i]: (x_smooth[i], y_smooth[i]) for i in range(len(indices))}
    return smoothed_centers


def calculate_crop_params(outputs_per_frame, video_frames, padding=20):
    """计算裁剪参数：保持原视频比例，平滑物体中心轨迹"""
    print("正在计算裁剪参数...")
    
    frame_height, frame_width = video_frames[0].shape[:2]
    aspect_ratio = frame_width / frame_height
    
    # 收集所有帧中物体的边界框和中心
    all_boxes = []
    frame_centers = {}  # {frame_idx: (center_x, center_y)}
    frame_sizes = {}    # {frame_idx: (width, height)}
    
    for frame_idx in sorted(outputs_per_frame.keys()):
        outputs = outputs_per_frame[frame_idx]
        
        if len(outputs['object_ids']) == 0:
            continue
        
        # 使用bbox（XYXY格式）而不是mask
        bbox = get_bounding_box_from_outputs(outputs)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            all_boxes.append(bbox)
            
            # 计算物体中心
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            frame_centers[frame_idx] = (center_x, center_y)
            
            # 记录物体尺寸
            obj_width = x_max - x_min
            obj_height = y_max - y_min
            frame_sizes[frame_idx] = (obj_width, obj_height)
    
    if not all_boxes:
        print("警告: 未检测到任何物体，使用整个视频")
        crop_width = frame_width
        crop_height = frame_height
        centers = {i: (frame_width / 2, frame_height / 2) for i in range(len(video_frames))}
        return crop_width, crop_height, centers
    
    # 找到所有帧中物体的最大尺寸
    all_boxes = np.array(all_boxes)
    widths = all_boxes[:, 2] - all_boxes[:, 0]
    heights = all_boxes[:, 3] - all_boxes[:, 1]
    
    # 使用最大的宽度和高度，加上padding
    max_obj_width = widths.max() + 2 * padding
    max_obj_height = heights.max() + 2 * padding
    
    # 根据原视频比例确定裁剪尺寸
    # 确保裁剪区域能包含最大的物体
    if aspect_ratio >= 1:  # 宽视频
        # 以高度为基准
        crop_height = max_obj_height
        crop_width = crop_height * aspect_ratio
        
        # 如果宽度不够，以宽度为基准
        if crop_width < max_obj_width:
            crop_width = max_obj_width
            crop_height = crop_width / aspect_ratio
    else:  # 高视频
        # 以宽度为基准
        crop_width = max_obj_width
        crop_height = crop_width / aspect_ratio
        
        # 如果高度不够，以高度为基准
        if crop_height < max_obj_height:
            crop_height = max_obj_height
            crop_width = crop_height * aspect_ratio
    
    # 确保裁剪尺寸不超过视频尺寸
    if crop_width > frame_width:
        crop_width = frame_width
        crop_height = crop_width / aspect_ratio
    if crop_height > frame_height:
        crop_height = frame_height
        crop_width = crop_height * aspect_ratio
    
    crop_width = int(crop_width)
    crop_height = int(crop_height)
    
    print(f"原视频尺寸: {frame_width} x {frame_height}")
    print(f"裁剪尺寸: {crop_width} x {crop_height} (保持比例 {aspect_ratio:.2f})")
    print(f"检测到物体的帧数: {len(frame_centers)}")
    
    # 对于没有检测到物体的帧，使用插值
    all_frame_centers = {}
    indices_with_detection = sorted(frame_centers.keys())
    
    for frame_idx in range(len(video_frames)):
        if frame_idx in frame_centers:
            all_frame_centers[frame_idx] = frame_centers[frame_idx]
        else:
            # 使用最近的有效帧
            if len(indices_with_detection) > 0:
                # 找到最近的检测帧
                closest_idx = min(indices_with_detection, key=lambda x: abs(x - frame_idx))
                all_frame_centers[frame_idx] = frame_centers[closest_idx]
            else:
                all_frame_centers[frame_idx] = (frame_width / 2, frame_height / 2)
    
    # 平滑物体中心轨迹，减少抖动
    print("正在平滑物体轨迹...")
    smoothed_centers = smooth_trajectory(all_frame_centers, window_size=16)
    
    return crop_width, crop_height, smoothed_centers


def create_output_videos(video_frames, outputs_per_frame, crop_width, crop_height, frame_centers, output_dir, fps, enable_crop=True):
    """创建两个输出视频：居中裁剪的原视频和白色背景版本"""
    frame_height, frame_width = video_frames[0].shape[:2]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件路径
    output_centered = os.path.join(output_dir, "centered_video.mp4")
    output_white_bg = os.path.join(output_dir, "white_background_video.mp4")
    
    centered_frames = []
    white_bg_frames = []
    
    print("正在生成输出视频...")
    for frame_idx in tqdm(range(len(video_frames)), desc="处理帧"):
        frame = video_frames[frame_idx]
        
        if enable_crop:
            # 启用裁剪模式
            # 获取物体中心
            center_x, center_y = frame_centers[frame_idx]
            
            # 计算裁剪区域，使物体居中
            half_width = crop_width // 2
            half_height = crop_height // 2
            crop_x_min = int(center_x - half_width)
            crop_y_min = int(center_y - half_height)
            crop_x_max = crop_x_min + crop_width
            crop_y_max = crop_y_min + crop_height
            
            # 调整裁剪区域，确保不超出视频边界（不露出黑边）
            if crop_x_min < 0:
                crop_x_max -= crop_x_min
                crop_x_min = 0
            if crop_y_min < 0:
                crop_y_max -= crop_y_min
                crop_y_min = 0
            if crop_x_max > frame_width:
                crop_x_min -= (crop_x_max - frame_width)
                crop_x_max = frame_width
            if crop_y_max > frame_height:
                crop_y_min -= (crop_y_max - frame_height)
                crop_y_max = frame_height
            
            # 再次确保边界
            crop_x_min = max(0, crop_x_min)
            crop_y_min = max(0, crop_y_min)
            crop_x_max = min(frame_width, crop_x_max)
            crop_y_max = min(frame_height, crop_y_max)
            
            # 裁剪帧
            cropped_frame = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy()
            
            # 如果裁剪后的尺寸不完全匹配（边界情况），调整大小
            if cropped_frame.shape[0] != crop_height or cropped_frame.shape[1] != crop_width:
                cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
            
            # 创建白色背景版本
            white_bg_frame = np.ones((crop_height, crop_width, 3), dtype=np.uint8) * 255
        else:
            # 不裁剪，使用原始帧
            cropped_frame = frame.copy()
            white_bg_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
            crop_y_min, crop_y_max = 0, frame_height
            crop_x_min, crop_x_max = 0, frame_width
        
        if frame_idx in outputs_per_frame:
            outputs = outputs_per_frame[frame_idx]
            
            if len(outputs['object_ids']) > 0:
                # 合并所有检测到的物体的mask
                combined_mask = np.zeros((frame_height, frame_width), dtype=bool)
                for i in range(len(outputs['object_ids'])):
                    mask = outputs['masks'][i].cpu().numpy()
                    combined_mask = combined_mask | mask
                
                if enable_crop:
                    # 裁剪mask
                    cropped_mask = combined_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                    
                    # 如果mask尺寸不匹配，调整大小
                    if cropped_mask.shape[0] != crop_height or cropped_mask.shape[1] != crop_width:
                        cropped_mask = cv2.resize(cropped_mask.astype(np.uint8), (crop_width, crop_height)) > 0.5
                    
                    # 将物体复制到白色背景上
                    white_bg_frame[cropped_mask] = cropped_frame[cropped_mask]
                else:
                    # 不裁剪，直接使用原始mask
                    white_bg_frame[combined_mask] = frame[combined_mask]
        
        centered_frames.append(cropped_frame)
        white_bg_frames.append(white_bg_frame)
    
    # 使用moviepy保存视频
    print("正在保存居中裁剪视频...")
    clip_centered = ImageSequenceClip(centered_frames, fps=fps)
    clip_centered.write_videofile(output_centered, codec='libx264', audio=False, logger=None)
    
    print("正在保存白色背景视频...")
    clip_white_bg = ImageSequenceClip(white_bg_frames, fps=fps)
    clip_white_bg.write_videofile(output_white_bg, codec='libx264', audio=False, logger=None)
    
    print(f"\n输出视频已保存:")
    print(f"  1. 居中裁剪视频: {output_centered}")
    print(f"  2. 白色背景视频: {output_white_bg}")
    
    return output_centered


def extrinsics_to_pose_format(extrinsics):
    """
    将Depth Anything 3的extrinsics (w2c, opencv格式) 转换为pose格式（OpenGL坐标系）
    
    Args:
        extrinsics: [3, 4] numpy array, opencv w2c格式
        
    Returns:
        dict: 包含x, y, z, rx, ry, rz的字典（OpenGL坐标系）
    """
    # extrinsics是w2c (world to camera)，需要转换为c2w (camera to world)
    R_w2c_cv = extrinsics[:3, :3]
    t_w2c_cv = extrinsics[:3, 3]
    
    # OpenCV -> OpenGL 坐标系转换矩阵
    # OpenCV: X右, Y下, Z前
    # OpenGL: X右, Y上, Z后
    # 转换: Y' = -Y, Z' = -Z
    cv_to_gl = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)
    
    # 将OpenCV的w2c转换为OpenGL的w2c
    R_w2c_gl = cv_to_gl @ R_w2c_cv @ cv_to_gl.T
    t_w2c_gl = cv_to_gl @ t_w2c_cv
    
    # 转换为c2w (OpenGL坐标系)
    R_c2w_gl = R_w2c_gl.T
    t_c2w_gl = -R_c2w_gl @ t_w2c_gl
    
    # 提取位置
    x, y, z = t_c2w_gl
    
    # 提取旋转（转换为欧拉角，单位：弧度）
    rotation = Rotation.from_matrix(R_c2w_gl)
    rx, ry, rz = rotation.as_euler('xyz', degrees=False)
    
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "rx": float(rx),
        "ry": float(ry),
        "rz": float(rz)
    }


def extract_camera_poses(video_path, output_dir, da3_model, device):
    """
    使用Depth Anything 3提取相机外参（OpenGL坐标系，相对第一帧）
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        da3_model: DA3模型名称
        device: 设备
        
    Note:
        - 输出使用OpenGL坐标系：X右, Y上, Z后（相机朝向-Z）
        - 外参相对于第一帧（第一帧为原点）
        - 帧数与输入视频对齐
    """
    print(f"\n正在提取相机外参（OpenGL坐标系，使用Depth Anything 3）...")
    
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        print("错误: 无法导入depth_anything_3，请安装: pip install depth-anything-3")
        return
    
    # 加载DA3模型
    print(f"加载Depth Anything 3模型: {da3_model}")
    da3 = DepthAnything3.from_pretrained(da3_model)
    da3 = da3.to(device=device)
    
    # 读取视频所有帧
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    
    # 提取所有帧
    frames = []
    for idx in tqdm(range(total_frames), desc="提取帧"):
        ret, frame = cap.read()
        if ret:
            # DA3需要RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break
    
    cap.release()
    
    actual_frames = len(frames)
    print(f"实际提取帧数: {actual_frames}")
    
    # 使用DA3推理
    print("正在使用Depth Anything 3推理...")
    prediction = da3.inference(frames)
    
    print(f"推理完成:")
    print(f"  - 图像: {prediction.processed_images.shape}")
    print(f"  - 深度: {prediction.depth.shape}")
    print(f"  - 置信度: {prediction.conf.shape}")
    print(f"  - 外参: {prediction.extrinsics.shape}")
    print(f"  - 内参: {prediction.intrinsics.shape}")
    
    # 获取第一帧的c2w作为参考（OpenGL坐标系）
    first_extrinsics = prediction.extrinsics[0]  # [3, 4]
    R_w2c_cv_0 = first_extrinsics[:3, :3]
    t_w2c_cv_0 = first_extrinsics[:3, 3]
    
    # OpenCV -> OpenGL 坐标系转换
    cv_to_gl = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)
    
    R_w2c_gl_0 = cv_to_gl @ R_w2c_cv_0 @ cv_to_gl.T
    t_w2c_gl_0 = cv_to_gl @ t_w2c_cv_0
    
    # 第一帧的c2w
    R_c2w_gl_0 = R_w2c_gl_0.T
    t_c2w_gl_0 = -R_c2w_gl_0 @ t_w2c_gl_0
    
    # 转换所有帧的外参为相对第一帧的格式
    ref_poses = []
    for i in range(len(prediction.extrinsics)):
        extrinsics = prediction.extrinsics[i]  # [3, 4]
        R_w2c_cv = extrinsics[:3, :3]
        t_w2c_cv = extrinsics[:3, 3]
        
        # 转换到OpenGL坐标系
        R_w2c_gl = cv_to_gl @ R_w2c_cv @ cv_to_gl.T
        t_w2c_gl = cv_to_gl @ t_w2c_cv
        
        # 转换为c2w
        R_c2w_gl = R_w2c_gl.T
        t_c2w_gl = -R_c2w_gl @ t_w2c_gl
        
        # 计算相对于第一帧的位姿
        # T_rel = T_0^{-1} @ T_i
        R_rel = R_c2w_gl_0.T @ R_c2w_gl
        t_rel = R_c2w_gl_0.T @ (t_c2w_gl - t_c2w_gl_0)
        
        # 提取位置
        x, y, z = t_rel
        
        # 提取旋转（转换为欧拉角，单位：弧度）
        rotation = Rotation.from_matrix(R_rel)
        rx, ry, rz = rotation.as_euler('xyz', degrees=False)
        
        pose = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "rx": float(rx),
            "ry": float(ry),
            "rz": float(rz)
        }
        ref_poses.append(pose)
    
    # 生成输出JSON
    video_name = Path(video_path).stem
    output_json = {
        "id": video_name,
        "num_frames": actual_frames,
        "ref": ref_poses
    }
    
    # 保存JSON
    json_path = os.path.join(output_dir, f"{video_name}_camera_poses.json")
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\n相机外参已保存到: {json_path}")
    print(f"包含 {len(ref_poses)} 个相机位姿（相对第一帧）")
    print(f"第一帧位姿: x=0, y=0, z=0, rx=0, ry=0, rz=0")


def main():
    args = parse_args()
    
    # 设置设备
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，使用CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(args.device)
            print(f"使用设备: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device(args.device)
        print(f"使用设备: {device}")
    
    # 检查是否是图片文件夹
    is_image_folder = os.path.isdir(args.video_path)
    
    # 创建输出目录：samples/sam3/视频名字/
    if is_image_folder:
        video_name = os.path.basename(args.video_path.rstrip('/'))
    else:
        video_name = Path(args.video_path).stem
    output_dir = os.path.join(args.output_dir, video_name)
    print(f"输出目录: {output_dir}")
    
    # 加载模型
    model, processor = load_sam3_model(args.model_name, device)
    
    # 加载视频或图片序列
    # 图片序列：fps=1，处理所有帧
    # 视频文件：使用process_fps采样
    video_frames, original_fps = load_video_frames(
        args.video_path, 
        max_short_side=480, 
        max_fps=16,
        process_fps=args.process_fps
    )
    
    # 确定输出fps
    fps = min(original_fps, 16)
    if args.fps is not None:
        fps = min(args.fps, 16)
    
    print(f"输出FPS: {fps:.2f}")
    print(f"追踪裁剪: {'启用' if args.enable_crop else '关闭'}")
    
    # 使用SAM3处理视频
    outputs_per_frame = process_video_with_sam3(
        model, processor, video_frames, args.text_prompt, device, args.max_frames
    )
    
    if args.enable_crop:
        # 计算裁剪参数（保持原视频比例，平滑轨迹）
        crop_width, crop_height, frame_centers = calculate_crop_params(
            outputs_per_frame, video_frames, args.padding
        )
        
        # 确保裁剪尺寸的短边也不超过480
        short_side = min(crop_width, crop_height)
        if short_side > 480:
            scale = 480 / short_side
            crop_width = int(crop_width * scale)
            crop_height = int(crop_height * scale)
            print(f"调整裁剪尺寸到: {crop_width} x {crop_height}")
    else:
        # 不裁剪，使用原始尺寸
        frame_height, frame_width = video_frames[0].shape[:2]
        crop_width = frame_width
        crop_height = frame_height
        frame_centers = {i: (frame_width / 2, frame_height / 2) for i in range(len(video_frames))}
        print("不裁剪模式：使用原始视频尺寸")
    
    # 创建输出视频
    output_centered_video = create_output_videos(
        video_frames, outputs_per_frame, crop_width, crop_height, frame_centers, 
        output_dir, fps, enable_crop=args.enable_crop
    )
    
    # 提取相机外参（如果启用）
    if args.extract_camera:
        extract_camera_poses(
            output_centered_video, 
            output_dir, 
            args.da3_model, 
            device
        )
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()

