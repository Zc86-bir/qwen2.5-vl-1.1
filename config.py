import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Determine OpenCV version and set up trackers accordingly
def create_tracker_factory():
    trackers = {}
    
    # Check OpenCV version
    major_ver, _, _ = cv2.__version__.split('.')
    major_ver = int(major_ver)
    
    if major_ver >= 4:  # OpenCV 4.x
        # Check if legacy module is available (for OpenCV 4.x)
        if hasattr(cv2, 'legacy'):
            # CSRT tracker - good accuracy but slower
            if hasattr(cv2.legacy, 'TrackerCSRT_create'):
                trackers['CSRT'] = cv2.legacy.TrackerCSRT_create
                
            # KCF tracker - good balance of speed and accuracy
            if hasattr(cv2.legacy, 'TrackerKCF_create'):
                trackers['KCF'] = cv2.legacy.TrackerKCF_create
                
            # MOSSE tracker - very fast but less accurate
            if hasattr(cv2.legacy, 'TrackerMOSSE_create'):
                trackers['MOSSE'] = cv2.legacy.TrackerMOSSE_create
                
            # MedianFlow tracker - good for predictable motion
            if hasattr(cv2.legacy, 'TrackerMedianFlow_create'):
                trackers['MedianFlow'] = cv2.legacy.TrackerMedianFlow_create
                
            # MIL tracker - good for handling occlusion
            if hasattr(cv2.legacy, 'TrackerMIL_create'):
                trackers['MIL'] = cv2.legacy.TrackerMIL_create
        else:
            logger.warning("OpenCV 4.x detected but legacy module not available")
            
    else:  # OpenCV 3.x
        # CSRT tracker - good accuracy but slower
        if hasattr(cv2, 'TrackerCSRT_create'):
            trackers['CSRT'] = cv2.TrackerCSRT_create
            
        # KCF tracker - good balance of speed and accuracy
        if hasattr(cv2, 'TrackerKCF_create'):
            trackers['KCF'] = cv2.TrackerKCF_create
            
        # MOSSE tracker - very fast but less accurate
        if hasattr(cv2, 'TrackerMOSSE_create'):
            trackers['MOSSE'] = cv2.TrackerMOSSE_create
            
        # MedianFlow tracker - good for predictable motion
        if hasattr(cv2, 'TrackerMedianFlow_create'):
            trackers['MedianFlow'] = cv2.TrackerMedianFlow_create
            
        # MIL tracker - good for handling occlusion
        if hasattr(cv2, 'TrackerMIL_create'):
            trackers['MIL'] = cv2.TrackerMIL_create
    
    # If no trackers are available, implement a basic fallback tracker
    if not trackers:
        logger.warning("No OpenCV trackers available, implementing basic tracker")
        trackers['Basic'] = create_basic_tracker
        
    return trackers

# Manual implementation of a basic tracker as a fallback
class BasicTracker:
    def __init__(self):
        self.bbox = None
        self.template = None
        self.initialized = False
    
    def init(self, frame, bbox):
        """Initialize the tracker with a bounding box"""
        x, y, w, h = [int(v) for v in bbox]
        self.bbox = (x, y, w, h)
        
        # Store the template for template matching
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:  # Check if ROI is valid
            return False
        
        # Convert to grayscale for template matching
        if len(frame.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        self.template = roi_gray
        self.initialized = True
        return True
    
    def update(self, frame):
        """Update tracker and return new bounding box"""
        if not self.initialized:
            return False, self.bbox
            
        x, y, w, h = self.bbox
        
        # Convert frame to grayscale
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        # Define search region (slightly larger than previous bbox)
        search_factor = 1.5
        search_x = max(0, int(x - w * (search_factor - 1) / 2))
        search_y = max(0, int(y - h * (search_factor - 1) / 2))
        search_w = min(frame.shape[1] - search_x, int(w * search_factor))
        search_h = min(frame.shape[0] - search_y, int(h * search_factor))
        
        if search_w <= 0 or search_h <= 0 or search_w < w or search_h < h:
            return False, self.bbox
        
        # Perform template matching in the search region
        search_region = frame_gray[search_y:search_y+search_h, search_x:search_x+search_w]
        if search_region.shape[0] < self.template.shape[0] or search_region.shape[1] < self.template.shape[1]:
            return False, self.bbox
            
        try:
            result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # Update bbox with new location
            new_x = search_x + max_loc[0]
            new_y = search_y + max_loc[1]
            
            # Ensure the bbox is within frame boundaries
            new_x = max(0, min(frame.shape[1] - w, new_x))
            new_y = max(0, min(frame.shape[0] - h, new_y))
            
            self.bbox = (new_x, new_y, w, h)
            return True, self.bbox
        except Exception as e:
            logger.error(f"Basic tracker error: {e}")
            return False, self.bbox

def create_basic_tracker():
    """Factory function for BasicTracker"""
    return BasicTracker()

# Create the tracking algorithms dictionary
TRACKING_ALGORITHMS = create_tracker_factory()

# Enhanced model configuration with additional parameters
SUPPORTED_MODELS = {
    "qvq-72b-preview": {
        "supports": ["text", "image", "video"],
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "qwen-omni-turbo": {
        "supports": ["text", "image", "video"],
        "max_tokens": 2048,
        "temperature": 0.8
    },
    "qwen-vl-ocr": {
        "supports": ["text", "image"],
        "max_tokens": 2048,
        "temperature": 0.5
    },
    "qwen-vl-max": {
        "supports": ["text", "image", "video"],
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "qwen2.5-vl-72b-instruct": {
        "supports": ["text", "image", "video"],
        "max_tokens": 4096,
        "temperature": 0.7
    },
    "qwen2.5-vl-7b-instruct": {
        "supports": ["text", "image", "video"],
        "max_tokens": 2048,
        "temperature": 0.7
    }
}

# Enhanced prompt templates
PROMPT_TEMPLATES = {
    "target_detection": """
<分析者角色>: 目标检测专家
<指令>: 请分析图像并识别其中的目标对象，返回每个目标的边界框（bbox_2d）和类别标签（label）。
<多媒体分析数据项>: 输出格式为JSON，包含字段 "annotations"，其中每个标注包括 "bbox_2d": [x1, y1, x2, y2] 和 "label": "类别"。
请确保所有坐标值都是整数，标注格式严格符合要求，以便系统自动处理。
""",
    "scene_analysis": """
<分析者角色>: 场景分析专家
<指令>: 请详细描述图像中的场景，包括主要对象、环境、氛围和关键细节。
<输出要求>: 提供结构化的场景描述，包括场景类型、主要元素和视觉特征。
""",
    "text_extraction": """
<分析者角色>: OCR专家
<指令>: 请提取图像中的所有文本内容，并保持原有格式和布局。
<输出要求>: 返回JSON格式，包含文本内容、位置信息和置信度。
""",
    "video_tracking": """
<分析者角色>: 视频分析与目标跟踪专家
<指令>: 请分析视频帧中的目标对象，返回每个目标的边界框（bbox_2d）、类别标签（label）和唯一ID（track_id）。
<输出要求>: 返回JSON格式，包含目标的位置、类别和跟踪ID信息，并确保同一目标在不同帧中的track_id一致。
"""
}

# 默认配置
DEFAULT_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "tracking_algorithm": "Basic",
    "tracking_interval": 5,  # 每5帧进行跟踪处理
    "visualization": {
        "show_labels": True,
        "show_scores": False,
        "color_scheme": "default"
    }
}

# 导入跟踪算法
try:
    from tracking_utils import TRACKING_ALGORITHMS
except ImportError:
    print("警告: 无法导入跟踪工具，将使用空字典")
    TRACKING_ALGORITHMS = {}