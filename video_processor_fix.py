"""
此文件提供跟踪器创建的修复实现，替换视频处理器中的跟踪器创建逻辑。
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

def create_video_tracker(algorithm="Basic"):
    """
    创建一个与 multimodal_app 兼容的视频跟踪器
    """
    from tracking_utils import create_tracker
    
    logger.info(f"视频处理器请求创建 {algorithm} 跟踪器")
    return create_tracker(algorithm)

def get_available_trackers():
    """
    获取可用的跟踪器列表
    """
    from tracking_utils import get_available_algorithms
    return get_available_algorithms()