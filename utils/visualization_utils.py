import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from datetime import datetime
import requests
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)

def get_font(size=20):
    """获取适合可视化文本的字体
    
    Args:
        size: 字体大小，默认为20
        
    Returns:
        适合显示的字体对象
    """
    # 字体选项，优先选择支持中文字符的字体
    font_options = [
        "simhei.ttf",     # 黑体 (Windows中文字体)
        "simsun.ttc",     # 宋体 (Windows中文字体)
        "msyh.ttc",       # 微软雅黑 (Windows中文字体)
        "NotoSansSC-Regular.otf",  # Noto Sans SC (谷歌CJK字体)
        "WenQuanYiMicroHei.ttf",   # 文泉驿微米黑 (Linux中文字体)
        "STHeiti Light.ttc",        # 黑体 (macOS中文字体)
        "PingFang.ttc",             # 苹方 (macOS中文字体)
        "arial.ttf",                # 回退到西方字体
        "DejaVuSans.ttf"
    ]
    
    # 尝试加载字体
    for font_name in font_options:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    
    # 如果找不到合适的字体，使用默认字体
    logger.warning("未找到合适的字体，使用默认字体")
    return ImageFont.load_default()

def get_color_scheme(scheme_name):
    """根据方案名称返回颜色调色板
    
    Args:
        scheme_name: 颜色方案名称
        
    Returns:
        颜色列表
    """
    schemes = {
        "rainbow": [  # 彩虹色
            (255, 0, 0), (255, 165, 0), (255, 255, 0), 
            (0, 255, 0), (0, 0, 255), (128, 0, 128)
        ],
        "pastel": [   # 柔和色
            (255, 179, 186), (255, 223, 186), (255, 255, 186),
            (186, 255, 201), (186, 225, 255), (218, 186, 255)
        ],
        "highvis": [  # 高可见度色
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ],
        "default": [  # 默认色
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255)
        ]
    }
    
    # 返回请求的方案，如不存在则返回默认方案
    return schemes.get(scheme_name, schemes["default"])

def visualize_detection(image, annotations, config):
    """为对象检测结果创建可视化
    
    Args:
        image: 输入图像（numpy数组、PIL图像或图像路径/URL）
        annotations: 检测注释列表，每项包含bbox_2d、label等信息
        config: 可视化配置字典
        
    Returns:
        添加了检测结果可视化的图像
    """
    try:
        # 如果输入是numpy数组，转换为PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            # 尝试从文件路径或URL加载
            if isinstance(image, str):
                if image.startswith(('http://', 'https://')):
                    response = requests.get(image)
                    image = Image.open(io.BytesIO(response.content))
                else:
                    image = Image.open(image)
        
        # 如果图像不是RGB模式，转换为RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        draw = ImageDraw.Draw(image)
        font = get_font(20)
        
        # 获取配置的颜色方案
        colors = get_color_scheme(config.get("color_scheme", "default"))
        class_color_map = {}  # 为每个类别保持一致的颜色
        
        # 按置信度排序注释（如果可用）
        sorted_annotations = sorted(
            annotations, 
            key=lambda x: float(x.get("confidence", 0.0) if x.get("confidence") else 0.0), 
            reverse=True
        )
        
        # 在图像上添加时间戳和模型信息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model = config.get("model", "未知")
        info_text = f"模型: {model} | 时间: {timestamp} | 目标数: {len(annotations)}"
        
        # 添加文字阴影提高可见度
        draw.text((11, 11), info_text, fill=(0, 0, 0), font=font)
        draw.text((10, 10), info_text, fill=(255, 255, 255), font=font)
        
        # 处理每个检测结果
        for i, anno in enumerate(sorted_annotations):
            bbox = anno.get("bbox_2d")
            label = anno.get("label", "未知")
            confidence = anno.get("confidence", None)
            track_id = anno.get("track_id", None)
            
            # 检查边界框是否有效
            if not bbox or len(bbox) != 4:
                continue
                
            # 为类别或跟踪ID分配一致的颜色
            if track_id is not None:
                # 使用track_id为跟踪保持一致的颜色
                if track_id not in class_color_map:
                    class_color_map[track_id] = colors[int(track_id) % len(colors)]
                color = class_color_map[track_id]
            else:
                # 使用类别标签为检测保持一致的颜色
                if label not in class_color_map:
                    class_color_map[label] = colors[len(class_color_map) % len(colors)]
                color = class_color_map[label]
            
            # 将坐标格式化为整数
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # 确保坐标在图像边界内
            width, height = image.size
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # 绘制更粗的边界框
            for thickness in range(3):
                draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], outline=color)
            
            # 准备显示文本，包含置信度和/或跟踪ID
            display_text = label
            if config.get("show_scores", False) and confidence is not None:
                conf_value = float(confidence) if isinstance(confidence, (int, float, str)) else 0.0
                display_text = f"{label}: {conf_value:.2f}"
                
            if track_id is not None:
                display_text = f"{display_text} [ID:{track_id}]"
            
            # 如果配置为显示标签，绘制文本
            if config.get("show_labels", True):
                # 创建文本背景
                text_size = draw.textbbox((0, 0), display_text, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                
                # 尽可能在框上方放置标签
                text_y = y1 - text_height - 2 if y1 > text_height + 2 else y1
                
                # 绘制文本背景
                draw.rectangle(
                    [x1, text_y, x1 + text_width + 6, text_y + text_height + 6],
                    fill=color
                )
                
                # 绘制文本
                draw.text((x1 + 3, text_y + 3), display_text, fill="white", font=font)
        
        return image
        
    except Exception as e:
        logger.error(f"可视化错误: {str(e)}")
        return None

def visualize_tracking(frame, tracks, track_history=None, config=None):
    """使用轨迹线可视化对象跟踪
    
    Args:
        frame: 输入帧图像
        tracks: 当前跟踪的目标字典 {track_id: (x, y, w, h)}
        track_history: 每个目标的历史轨迹字典 {track_id: [(x, y, w, h), ...]}
        config: 可视化配置字典
        
    Returns:
        添加了跟踪可视化的帧
    """
    if config is None:
        config = {}
        
    # 创建帧的副本以便绘图
    vis_frame = frame.copy()
    
    # 获取可视化颜色
    colors = get_color_scheme(config.get("color_scheme", "default"))
    
    # 将帧转换为PIL图像以获得更好的文本渲染
    frame_pil = Image.fromarray(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font = get_font(16)
    
    # 绘制当前边界框
    for track_id, bbox in tracks.items():
        # 选择颜色
        color_idx = int(track_id) % len(colors)
        color_rgb = colors[color_idx]
        # OpenCV使用BGR顺序
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        x, y, w, h = bbox
        # 绘制矩形
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color_bgr, 2)
        
        # 添加带有跟踪ID的标签
        label = f"ID: {track_id}"
        
        # 添加带背景的标签文本
        text_size = draw.textbbox((0, 0), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        
        # 绘制文本背景
        draw.rectangle(
            [x, y - text_height - 2, x + text_width + 6, y],
            fill=color_rgb
        )
        
        # 绘制文本
        draw.text((x + 3, y - text_height), label, fill="white", font=font)
    
    # 如果提供了跟踪历史，绘制轨迹线
    if track_history:
        for track_id, history in track_history.items():
            # 至少需要两个点才能绘制线
            if len(history) < 2:
                continue
                
            color_idx = int(track_id) % len(colors)
            color_bgr = (colors[color_idx][2], colors[color_idx][1], colors[color_idx][0])
            
            # 绘制连接历史点的线
            for i in range(1, len(history)):
                pt1 = (history[i-1][0] + history[i-1][2]//2, history[i-1][1] + history[i-1][3]//2)
                pt2 = (history[i][0] + history[i][2]//2, history[i][1] + history[i][3]//2)
                cv2.line(vis_frame, pt1, pt2, color_bgr, 2)
    
    # 将PIL图像转换回numpy数组/OpenCV格式
    vis_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    return vis_frame

def create_video_mosaic(frames, titles=None, grid_size=None):
    """创建多个视频帧的马赛克
    
    Args:
        frames: 要组合的帧列表
        titles: 每个帧的标题列表
        grid_size: 网格大小元组 (宽度, 高度)
        
    Returns:
        包含所有帧的马赛克图像
    """
    if not frames:
        return None
        
    # 如果未提供网格大小，自动确定
    if grid_size is None:
        n_frames = len(frames)
        grid_width = int(np.ceil(np.sqrt(n_frames)))
        grid_height = int(np.ceil(n_frames / grid_width))
        grid_size = (grid_width, grid_height)
    else:
        grid_width, grid_height = grid_size
    
    # 通过调整大小确保所有帧具有相同的尺寸
    frame_h, frame_w = frames[0].shape[:2]
    for i in range(1, len(frames)):
        if frames[i].shape[:2] != (frame_h, frame_w):
            frames[i] = cv2.resize(frames[i], (frame_w, frame_h))
    
    # 创建空的马赛克画布
    mosaic = np.zeros((frame_h * grid_height, frame_w * grid_width, 3), dtype=np.uint8)
    
    # 填充马赛克
    frame_idx = 0
    for i in range(grid_height):
        for j in range(grid_width):
            if frame_idx < len(frames):
                # 计算当前帧在马赛克中的位置
                y_start = i * frame_h
                y_end = y_start + frame_h
                x_start = j * frame_w
                x_end = x_start + frame_w
                
                # 放置当前帧
                mosaic[y_start:y_end, x_start:x_end] = frames[frame_idx]
                
                # 如果提供了标题，添加标题
                if titles and frame_idx < len(titles):
                    title = titles[frame_idx]
                    cv2.putText(mosaic, title, 
                               (x_start + 10, y_start + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                frame_idx += 1
            else:
                break
    
    return mosaic