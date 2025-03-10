import cv2
import numpy as np
import logging
import sys
import os
import io
import base64
import requests
import json
import time
import threading
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, Any, Tuple, List, Union
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import deque

# 配置日志
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# YOLO相关配置
DEFAULT_YOLO_MODEL = "yolov12s.pt"  # 默认使用中型模型，平衡速度和精度
YOLO_MODEL_PATHS = {
    "yolov12n": "models/yolov12n.pt",
    "yolov12s": "models/yolov12s.pt",
    "yolov12m": "models/yolov12m.pt",
    "yolov12l": "models/yolov12l.pt",
    "yolov12x": "models/yolov12x.pt",
}

# 创建支持重试和连接池的请求会话
def create_retry_session(retries=3, backoff_factor=0.3, 
                         status_forcelist=(500, 502, 504), 
                         session=None):
    """创建一个带有重试功能的请求会话"""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class YOLOv12Detector:
    """YOLOv12对象检测器类"""
    
    def __init__(self, model_name="yolov12s", conf_thres=0.25, iou_thres=0.45, device=""):
        """初始化YOLOv12检测器
        
        Args:
            model_name: 模型名称，可选: "yolov12n", "yolov12s", "yolov12m", "yolov12l", "yolov12x"
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
            device: 运行设备，空字符串表示自动选择
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.device = device
        self.model = None
        self.model_name = model_name
        self.class_names = []
        self.chinese_names = {}  # 中文类别名称映射
        self._load_chinese_names()
        self.initialized = False
        self.width = 640  # 默认输入尺寸
        self.height = 640
    
    def _load_chinese_names(self):
        """加载类别的中文名称"""
        # COCO数据集的80个类别的中文映射，以及增加的动物标签
        self.chinese_names = {
            'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车',
            'airplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车',
            'boat': '船', 'traffic light': '红绿灯', 'fire hydrant': '消防栓',
            'stop sign': '停止标志', 'parking meter': '停车计时器', 'bench': '长椅',
            'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马',
            'sheep': '羊', 'cow': '牛', 'elephant': '大象', 'bear': '熊',
            'zebra': '斑马', 'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞',
            'handbag': '手提包', 'tie': '领带', 'suitcase': '手提箱', 'frisbee': '飞盘',
            'skis': '滑雪板', 'snowboard': '单板滑雪', 'sports ball': '运动球',
            'kite': '风筝', 'baseball bat': '棒球棒', 'baseball glove': '棒球手套',
            'skateboard': '滑板', 'surfboard': '冲浪板', 'tennis racket': '网球拍',
            'bottle': '瓶子', 'wine glass': '酒杯', 'cup': '杯子', 'fork': '叉子',
            'knife': '刀', 'spoon': '勺子', 'bowl': '碗', 'banana': '香蕉',
            'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子', 'broccoli': '西兰花',
            'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '披萨', 'donut': '甜甜圈',
            'cake': '蛋糕', 'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽',
            'bed': '床', 'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视',
            'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器', 'keyboard': '键盘',
            'cell phone': '手机', 'microwave': '微波炉', 'oven': '烤箱', 'toaster': '烤面包机',
            'sink': '水槽', 'refrigerator': '冰箱', 'book': '书', 'clock': '时钟',
            'vase': '花瓶', 'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机',
            'toothbrush': '牙刷','bicycle wheel': '自行车轮', 'motorcycle wheel': '摩托车轮', 'car wheel': '汽车轮',
            'license plate': '车牌', 'traffic sign': '交通标志', 'street light': '路灯',
            'sidewalk': '人行道', 'road': '道路', 'crosswalk': '人行横道',
            'building': '建筑', 'house': '房子', 'apartment': '公寓', 'skyscraper': '摩天大楼',
            'bridge': '桥', 'tunnel': '隧道', 'tower': '塔',
            'tree': '树', 'flower': '花', 'grass': '草', 'plant': '植物', 'bush': '灌木',
            'forest': '森林', 'mountain': '山', 'hill': '小山', 'rock': '岩石',
            'river': '河流', 'lake': '湖', 'sea': '海', 'ocean': '大海', 'beach': '海滩',
            'waterfall': '瀑布', 'desert': '沙漠', 'snow': '雪', 'ice': '冰',
            'sky': '天空', 'cloud': '云', 'sun': '太阳', 'moon': '月亮', 'star': '星星',
            'phone': '电话', 'computer': '电脑', 'screen': '屏幕', 'monitor': '显示器',
            'desk': '桌子', 'table': '桌子', 'cabinet': '柜子', 'drawer': '抽屉',
            'door': '门', 'window': '窗户', 'curtain': '窗帘', 'blind': '百叶窗',
            'wall': '墙', 'floor': '地板', 'ceiling': '天花板', 'roof': '屋顶',
            'stair': '楼梯', 'elevator': '电梯', 'escalator': '自动扶梯',
            'food': '食物', 'fruit': '水果', 'vegetable': '蔬菜', 'meat': '肉',
            'fish': '鱼', 'bread': '面包', 'rice': '米饭', 'noodle': '面条',
            'drink': '饮料', 'water': '水', 'coffee': '咖啡', 'tea': '茶', 'milk': '牛奶',
            'backpack': '背包', 'handbag': '手提包', 'wallet': '钱包', 'purse': '钱包',
            'helmet': '头盔', 'hat': '帽子', 'cap': '鸭舌帽', 'glasses': '眼镜', 'sunglasses': '太阳镜',
            'shoe': '鞋', 'boot': '靴子', 'sandal': '凉鞋', 'slipper': '拖鞋',
            'clothes': '衣服', 'shirt': '衬衫', 'jacket': '夹克', 'coat': '外套', 'pants': '裤子',
            'dress': '连衣裙', 'skirt': '裙子', 'sock': '袜子', 'glove': '手套',
            'baby': '婴儿', 'child': '儿童', 'boy': '男孩', 'girl': '女孩',
            'man': '男人', 'woman': '女人', 'student': '学生', 'teacher': '老师',
            'worker': '工人', 'driver': '司机', 'police': '警察', 'soldier': '军人',
            'face': '脸', 'eye': '眼睛', 'nose': '鼻子', 'mouth': '嘴巴', 'ear': '耳朵',
            'hair': '头发', 'hand': '手', 'foot': '脚', 'leg': '腿', 'arm': '手臂',
            'tiger': '老虎', 'lion': '狮子', 'leopard': '豹子', 'cheetah': '猎豹',
            'wolf': '狼', 'fox': '狐狸', 'raccoon': '浣熊', 'deer': '鹿',
            'moose': '驼鹿', 'elk': '麋鹿', 'reindeer': '驯鹿', 'antelope': '羚羊',
            'buffalo': '水牛', 'bison': '野牛', 'rhino': '犀牛', 'hippo': '河马',
            'giraffe': '长颈鹿', 'camel': '骆驼', 'llama': '羊驼', 'alpaca': '羊驼',
            'panda': '熊猫', 'koala': '考拉', 'kangaroo': '袋鼠', 'sloth': '树懒',
            'monkey': '猴子', 'gorilla': '大猩猩', 'chimpanzee': '黑猩猩', 'orangutan': '猩猩',
            'crocodile': '鳄鱼', 'alligator': '短吻鳄', 'turtle': '乌龟', 'tortoise': '陆龟',
            'snake': '蛇', 'lizard': '蜥蜴', 'gecko': '壁虎', 'iguana': '鬣蜥',
            'frog': '青蛙', 'toad': '蟾蜍', 'newt': '蝾螈', 'salamander': '火蜥蜴',
            'fish': '鱼', 'shark': '鲨鱼', 'dolphin': '海豚', 'whale': '鲸鱼',
            'octopus': '章鱼', 'squid': '鱿鱼', 'jellyfish': '水母', 'starfish': '海星',
            'penguin': '企鹅', 'seagull': '海鸥', 'pelican': '鹈鹕', 'swan': '天鹅',
            'goose': '鹅', 'duck': '鸭子', 'chicken': '鸡', 'turkey': '火鸡',
            'parrot': '鹦鹉', 'eagle': '鹰', 'hawk': '鹰', 'falcon': '隼',
            'owl': '猫头鹰', 'pigeon': '鸽子', 'dove': '鸽子', 'sparrow': '麻雀',
            'butterfly': '蝴蝶', 'moth': '蛾', 'bee': '蜜蜂', 'wasp': '黄蜂',
            'spider': '蜘蛛', 'scorpion': '蝎子', 'ant': '蚂蚁', 'beetle': '甲虫',
            'cricket': '蟋蟀', 'grasshopper': '蚱蜢', 'dragonfly': '蜻蜓', 'mosquito': '蚊子',
            'crab': '螃蟹', 'lobster': '龙虾', 'crawfish': '小龙虾', 'shrimp': '虾',
            'rabbit': '兔子', 'hare': '野兔', 'hamster': '仓鼠', 'guinea pig': '豚鼠',
            'mouse': '老鼠', 'rat': '大鼠', 'hedgehog': '刺猬', 'squirrel': '松鼠',
            'bat': '蝙蝠', 'armadillo': '犰狳', 'opossum': '负鼠', 'otter': '水獭',
            'weasel': '黄鼬', 'ferret': '雪貂', 'mink': '水貂', 'badger': '獾',
            'skunk': '臭鼬', 'mole': '鼹鼠', 'goat': '山羊', 'donkey': '驴',
            'mule': '骡子', 'pig': '猪', 'boar': '野猪', 'racoon': '浣熊',
        }
    
    def initialize(self):
        """初始化模型"""
        if self.initialized:
            return True
        
        try:
            # 检查是否已安装ultralytics
            import importlib.util
            if importlib.util.find_spec("ultralytics") is None:
                logger.error("请安装ultralytics库: pip install ultralytics")
                return False
            
            from ultralytics import YOLO
            
            # 获取模型路径
            model_path = YOLO_MODEL_PATHS.get(self.model_name, None)
            if not model_path:
                logger.error(f"未找到模型 {self.model_name} 的路径")
                return False
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 加载模型
            logger.info(f"正在加载YOLOv12模型: {model_path}")
            self.model = YOLO(model_path)
            
            # 获取类别名称
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
            
            logger.info(f"YOLOv12模型加载成功，类别数: {len(self.class_names)}")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"初始化YOLOv12模型时出错: {e}")
            return False
    
    def detect(self, frame):
        """检测图像中的对象
        
        Args:
            frame: OpenCV格式的图像
            
        Returns:
            字典列表，每个字典包含检测结果信息
        """
        if not self.initialized and not self.initialize():
            logger.error("模型未初始化")
            return []
        
        try:
            # 运行推理
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]  # 取第一个结果
                
                # 提取边界框、类别和置信度
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        # 提取坐标 (XYXY格式)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # 转换为XYWH格式
                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)
                        
                        # 类别和置信度
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        
                        # 获取类别名称
                        cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        
                        # 获取中文类别名称
                        chinese_name = self.chinese_names.get(cls_name, cls_name)
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'class_id': cls_id,
                            'class_name': cls_name, 
                            'chinese_name': chinese_name,
                            'confidence': conf
                        })
            
            return detections
                
        except Exception as e:
            logger.error(f"YOLOv12检测出错: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """在图像上绘制检测结果
        
        Args:
            frame: OpenCV格式的图像
            detections: 检测结果列表
            
        Returns:
            绘制了检测结果的图像
        """
        result_frame = frame.copy()
        
        for det in detections:
            x, y, w, h = det['bbox']
            cls_name = det['chinese_name']  # 使用中文类别名称
            conf = det['confidence']
            
            # 不同类别使用不同颜色
            color = self._get_color(det['class_id'])
            
            # 绘制边界框
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制带有类别名称和置信度的标签
            label = f"{cls_name} {conf:.2f}"
            result_frame = draw_text_on_image(
                result_frame,
                label,
                position=(x, y - 10),
                font_size=15,
                color=(255, 255, 255),
                bg_color=color,
            )
        
        return result_frame
    
    def _get_color(self, class_id):
        """根据类别ID获取颜色
        
        Args:
            class_id: 类别ID
            
        Returns:
            颜色元组 (B, G, R)
        """
        # 使用一组预定义的颜色
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 粉色
            (0, 165, 255),  # 橙色
            (128, 0, 128),  # 紫色
            (255, 192, 203),# 粉红色
            (128, 128, 0)   # 橄榄色
        ]
        
        return colors[class_id % len(colors)]


# 大模型API客户端类
class ModelAPIClient:
    """与大模型API交互的客户端"""
    
    def __init__(self, api_key=None, model_name="qwen-vl-max", timeout=20):
        """初始化API客户端
        
        Args:
            api_key: API密钥（如果为None则尝试从环境变量获取）
            model_name: 模型名称
            timeout: 请求超时时间（秒）
        """
        # 获取API密钥
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            logger.warning("未找到API密钥，请设置DASHSCOPE_API_KEY环境变量或在初始化时提供")
        
        self.model_name = model_name
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.timeout = timeout
        self.session = create_retry_session()  # 创建带有重试功能的会话
        
        # 离线标签库（当API不可用时使用）
        self.offline_labels = ["人", "车", "狗", "猫", "鸟", "椅子", "桌子", "手机", "电脑", "树木", "房屋"]
        
        logger.info(f"初始化模型API客户端，使用模型: {self.model_name}")
    
    def analyze_image(self, image: Image.Image, prompt: str = "这张图片中有什么？请识别主要物体并给出简短的标签。") -> str:
        """分析图像内容，返回识别结果
        
        Args:
            image: PIL图像对象
            prompt: 提示词
            
        Returns:
            识别结果文本
        """
        if not self.api_key:
            return "未设置API密钥"
            
        try:
            # 调整图像大小，减少数据传输量
            max_dim = 256  # 更小的尺寸以避免网络问题
            if max(image.width, image.height) > max_dim:
                if image.width > image.height:
                    new_width = max_dim
                    new_height = int(image.height * (max_dim / image.width))
                else:
                    new_height = max_dim
                    new_width = int(image.width * (max_dim / image.height))
                image = image.resize((new_width, new_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
            # 将图像编码为base64
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=70)  # 降低质量以减小数据量
            img_bytes = buffer.getvalue()
            base64_data = base64.b64encode(img_bytes).decode("utf-8")
            img_data_uri = f"data:image/jpeg;base64,{base64_data}"
            
            # 构建请求
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"text": "你是一个视觉识别助手，请简洁地识别图片中的主要物体，用10个字以内的中文标签表示。"}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        {"image": img_data_uri}
                    ]
                }
            ]
            
            # 构建请求数据
            request_data = {
                "model": self.model_name,
                "input": {
                    "messages": messages
                },
                "parameters": {}
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 发送请求，使用会话和超时设置
            response = self.session.post(
                self.base_url,
                json=request_data,
                headers=headers,
                timeout=self.timeout
            )
            
            # 处理错误
            if response.status_code != 200:
                logger.error(f"API返回错误: {response.status_code}, 响应: {response.text}")
                return self._get_fallback_label(image)  # 使用备选方案
            
            # 解析响应
            result = response.json()
            
            try:
                if "output" in result and "choices" in result["output"] and len(result["output"]["choices"]) > 0:
                    choice = result["output"]["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        for content_item in choice["message"]["content"]:
                            if "text" in content_item:
                                # 提取并清理识别结果
                                text = content_item["text"].strip()
                                # 如果结果过长，只取前10个字符
                                if len(text) > 10:
                                    text = text[:10].strip()
                                return text
                                
                return self._get_fallback_label(image)  # 使用备选方案
            except Exception as e:
                logger.error(f"解析API响应时出错: {e}")
                return self._get_fallback_label(image)  # 使用备选方案
                
        except requests.exceptions.SSLError as e:
            logger.error(f"SSL错误: {e}")
            return self._get_fallback_label(image)  # SSL错误时使用备选方案
        except requests.exceptions.ConnectionError as e:
            logger.error(f"连接错误: {e}")
            return self._get_fallback_label(image)  # 连接错误时使用备选方案
        except requests.exceptions.Timeout as e:
            logger.error(f"请求超时: {e}")
            return self._get_fallback_label(image)  # 超时时使用备选方案
        except Exception as e:
            logger.error(f"调用API时出错: {e}")
            return self._get_fallback_label(image)  # 其他错误时使用备选方案
    
    def _get_fallback_label(self, image=None) -> str:
        """当API调用失败时获取备选标签"""
        if image is not None:
            # 这里可以实现简单的本地图像分析逻辑
            # 例如检测图像亮度、颜色分布等特征
            try:
                # 简单的亮度分析
                img_array = np.array(image)
                brightness = np.mean(img_array)
                
                # 根据亮度选择不同的标签
                if brightness > 200:
                    return "明亮物体"
                elif brightness < 50:
                    return "暗色物体"
                else:
                    # 随机选择一个标签
                    import random
                    return random.choice(self.offline_labels)
            except:
                pass
        
        # 默认返回
        return "未知目标"


# 中文字体相关功能
class FontManager:
    """字体管理器，用于加载和管理中文字体"""
    
    def __init__(self):
        self.default_font = None
        self.fonts = {}
        self.init_fonts()
    
    def init_fonts(self):
        """初始化字体"""
        # 字体搜索路径
        font_paths = [
            # Windows 字体路径
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simsun.ttc',
            # macOS 字体路径
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            # Linux 字体路径
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
            # 当前目录下的字体
            './fonts/simhei.ttf',
            './fonts/msyh.ttc',
        ]
        
        # 尝试加载字体
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.default_font = ImageFont.truetype(path, 20)
                    logger.info(f"已加载中文字体: {path}")
                    return
                except Exception as e:
                    logger.warning(f"无法加载字体 {path}: {e}")
        
        # 如果没有找到中文字体，使用默认字体
        logger.warning("未找到中文字体，将使用默认字体")
        self.default_font = ImageFont.load_default()
    
    def get_font(self, size=20, name=None):
        """获取指定大小的字体"""
        if name is None:
            # 默认字体的不同大小
            if size not in self.fonts:
                if self.default_font is None:
                    self.init_fonts()
                
                try:
                    # 尝试获取默认字体的字体文件路径
                    font_path = self.default_font.path
                    self.fonts[size] = ImageFont.truetype(font_path, size)
                except (AttributeError, OSError):
                    # 如果无法获取路径或加载失败，使用默认字体
                    logger.warning(f"无法创建大小为 {size} 的字体，使用默认字体")
                    self.fonts[size] = self.default_font
            
            return self.fonts[size]
        else:
            # 未实现其他字体的加载，总是返回默认字体
            logger.warning(f"未实现字体 {name} 的加载，使用默认字体")
            return self.get_font(size)

# 创建全局字体管理器
font_manager = FontManager()

def draw_text_on_image(image, text, position, font_size=20, color=(255, 255, 255), bg_color=None, outline=True):
    """在图像上绘制中文文本"""
    # OpenCV图像转换为PIL图像
    if len(image.shape) == 3:
        # BGR转RGB
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        # 灰度图
        pil_img = Image.fromarray(image)
    
    # 创建绘图对象
    draw = ImageDraw.Draw(pil_img)
    
    # 获取字体
    font = font_manager.get_font(size=font_size)
    
    # 获取文本大小 - 兼容各版本Pillow
    try:
        # 最新版本的Pillow (9.0.0+)
        if hasattr(font, 'getbbox'):
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        # 中间版本 - 使用getsize
        elif hasattr(font, 'getsize'):
            text_width, text_height = font.getsize(text)
        # 旧版本 - 使用draw.textsize
        elif hasattr(draw, 'textsize'):
            text_width, text_height = draw.textsize(text, font=font)
        else:
            # 如果都不支持，使用字体大小的估计值
            text_width = len(text) * font_size * 0.6
            text_height = font_size * 1.2
            logger.warning(f"无法获取文本大小，使用估计值: {text_width}x{text_height}")
    except Exception as e:
        # 出错时使用估计值
        text_width = len(text) * font_size * 0.6
        text_height = font_size * 1.2
        logger.warning(f"获取文本大小时出错: {e}，使用估计值: {text_width}x{text_height}")
    
    x, y = position
    
    # 绘制背景矩形
    if bg_color:
        draw.rectangle([x, y, x + text_width, y + text_height], fill=bg_color)
    
    # 绘制文字描边/轮廓 (如果需要)
    if outline:
        outline_color = (0, 0, 0) if color[0] > 128 else (255, 255, 255)
        for offset_x, offset_y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            draw.text((x + offset_x, y + offset_y), text, font=font, fill=outline_color)
    
    # 绘制文本
    draw.text((x, y), text, font=font, fill=color)
    
    # PIL图像转回OpenCV图像
    if len(image.shape) == 3:
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        result_img = np.array(pil_img)
    
    return result_img


class YOLOTracker:
    """基于YOLOv12的智能对象跟踪器"""
    
    def __init__(self, detector=None, api_client=None, track_history=30):
        """初始化YOLOv12跟踪器
        
        Args:
            detector: YOLOv12检测器实例，如果为None则自动创建
            api_client: 大模型API客户端，如果为None则使用检测器的本地分类
            track_history: 轨迹历史长度
        """
        # 初始化检测器
        self.detector = detector or YOLOv12Detector()
        if not self.detector.initialized:
            self.detector.initialize()
        
        self.api_client = api_client
        self.tracks = {}  # 跟踪记录: {track_id: {bbox, class_id, class_name, ...}}
        self.next_id = 0  # 下一个跟踪ID
        self.track_history_len = track_history  # 轨迹历史长度
        self.lost_tracks = {}  # 丢失的跟踪记录，保留一段时间以便恢复
        self.max_lost_frames = 30  # 最大丢失帧数
        self.detection_interval = 5  # 每隔多少帧进行一次完整检测
        self.frame_count = 0  # 帧计数器
        
        # 跟踪参数
        self.iou_threshold = 0.3  # IOU阈值，用于匹配检测结果和跟踪记录
        self.confidence_threshold = 0.3  # 置信度阈值
        
        # 当前选中的跟踪目标
        self.selected_id = None
        self.label = "目标"  # 当前选中目标的标签
        self.bbox = None  # 当前选中目标的边界框
        
        logger.info("初始化YOLO跟踪器")
    
    def init(self, frame, bbox, label=None):
        """初始化跟踪器
        
        Args:
            frame: 视频帧
            bbox: 边界框 (x, y, w, h)
            label: 目标标签，如果为None则尝试自动识别
        """
        # 清空现有跟踪记录
        self.tracks = {}
        self.lost_tracks = {}
        self.next_id = 0
        self.frame_count = 0
        
        x, y, w, h = [int(v) for v in bbox]
        self.bbox = (x, y, w, h)
        
        # 如果提供了标签，直接使用
        if label:
            self.label = label
        
        # 添加初始跟踪记录
        track_id = self.next_id
        self.next_id += 1
        
        # 尝试使用YOLO检测器识别目标
        detections = self.detector.detect(frame)
        matched_detection = None
        
        # 找到与初始边界框重叠最大的检测结果
        best_iou = 0
        for det in detections:
            det_bbox = det['bbox']
            iou = self._calculate_iou(bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                matched_detection = det
        
        # 如果找到匹配的检测结果，使用其类别信息
        if matched_detection and best_iou > 0.5:
            class_id = matched_detection['class_id']
            class_name = matched_detection['class_name']
            chinese_name = matched_detection['chinese_name']
            
            # 如果未提供标签，使用检测的中文类别名称
            if not label:
                self.label = chinese_name
        else:
            # 否则使用API客户端进行识别
            if not label and self.api_client:
                # 裁剪目标区域
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:  # 确保ROI有效
                    # 转换为PIL图像
                    pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    # 使用API识别
                    self.label = self.api_client.analyze_image(pil_img)
                    logger.info(f"API识别结果: {self.label}")
        
        # 添加跟踪记录
        self.tracks[track_id] = {
            'bbox': self.bbox,
            'label': self.label,
            'age': 0,  # 跟踪已存在的帧数
            'missed': 0,  # 连续丢失的帧数
            'trajectory': deque(maxlen=self.track_history_len),  # 轨迹历史
            'last_detection': self.bbox  # 最后一次检测到的边界框
        }
        
        # 添加当前位置到轨迹
        center_x = x + w // 2
        center_y = y + h // 2
        self.tracks[track_id]['trajectory'].append((center_x, center_y))
        
        # 选择当前跟踪目标
        self.selected_id = track_id
        
        logger.info(f"初始化跟踪器，选择目标ID: {track_id}, 标签: {self.label}")
        
        return True
    
    def update(self, frame):
        """更新跟踪器状态
        
        Args:
            frame: 当前视频帧
            
        Returns:
            (success, bbox): 跟踪成功标志和边界框
        """
        self.frame_count += 1
        
        # 定期检测
        if self.frame_count % self.detection_interval == 0:
            # 调用YOLO检测器获取当前帧的所有对象
            detections = self.detector.detect(frame)
            
            # 匹配检测结果与现有跟踪记录
            self._match_detections(detections)
        
        # 更新所有跟踪记录
        active_tracks = {}
        for track_id, track in self.tracks.items():
            # 更新年龄
            track['age'] += 1
            
            # 使用运动预测更新位置（如果未在本帧检测到）
            if track['missed'] > 0:
                # 简单的线性预测
                if len(track['trajectory']) >= 2:
                    last_point = track['trajectory'][-1]
                    prev_point = track['trajectory'][-2]
                    dx = last_point[0] - prev_point[0]
                    dy = last_point[1] - prev_point[1]
                    
                    x, y, w, h = track['bbox']
                    new_x = max(0, x + dx)
                    new_y = max(0, y + dy)
                    
                    # 确保边界框在图像范围内
                    frame_h, frame_w = frame.shape[:2]
                    new_x = min(new_x, frame_w - w)
                    new_y = min(new_y, frame_h - h)
                    
                    track['bbox'] = (new_x, new_y, w, h)
                
                track['missed'] += 1
            
            # 添加当前位置到轨迹
            x, y, w, h = track['bbox']
            center_x = x + w // 2
            center_y = y + h // 2
            track['trajectory'].append((center_x, center_y))
            
            # 检查是否丢失太久
            if track['missed'] <= self.max_lost_frames:
                active_tracks[track_id] = track
            else:
                self.lost_tracks[track_id] = track
        
        # 更新跟踪记录
        self.tracks = active_tracks
        
        # 如果没有选择的跟踪目标，或者选择的目标丢失，选择一个新的
        if self.selected_id is None or self.selected_id not in self.tracks:
            # 尝试从现有跟踪中选择一个
            if self.tracks:
                # 选择最接近中心的目标
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                best_distance = float('inf')
                best_id = None
                
                for track_id, track in self.tracks.items():
                    x, y, w, h = track['bbox']
                    track_center_x = x + w // 2
                    track_center_y = y + h // 2
                    
                    distance = ((track_center_x - center_x) ** 2 + 
                                (track_center_y - center_y) ** 2) ** 0.5
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_id = track_id
                
                if best_id is not None:
                    self.selected_id = best_id
                    self.label = self.tracks[best_id]['label']
                    self.bbox = self.tracks[best_id]['bbox']
                else:
                    self.selected_id = None
                    self.bbox = None
                    return False, (0, 0, 0, 0)
            else:
                self.selected_id = None
                self.bbox = None
                return False, (0, 0, 0, 0)
        
        # 返回选中目标的边界框
        if self.selected_id in self.tracks:
            self.bbox = self.tracks[self.selected_id]['bbox']
            return True, self.bbox
        else:
            return False, (0, 0, 0, 0)
    
    def draw_bbox(self, frame, color=(0, 255, 0)):
        """在帧上绘制边界框和标签"""
        if self.bbox is None:
            return frame
            
        result_frame = frame.copy()
        
        # 绘制所有跟踪的目标（使用淡色）
        for track_id, track in self.tracks.items():
            if track_id != self.selected_id:  # 不是选中的目标
                x, y, w, h = track['bbox']
                label = track.get('label', '目标')
                
                # 使用淡色绘制非选中目标
                faded_color = (192, 192, 192)  # 灰色
                cv2.rectangle(result_frame, (int(x), int(y)), 
                             (int(x + w), int(y + h)), faded_color, 2)
                
                # 绘制轨迹
                if 'trajectory' in track and len(track['trajectory']) > 1:
                    traj = np.array(track['trajectory'], dtype=np.int32)
                    for i in range(1, len(traj)):
                        cv2.line(result_frame, tuple(traj[i-1]), tuple(traj[i]), 
                                faded_color, 2)
        
        # 绘制选中的目标（高亮）
        if self.selected_id is not None and self.selected_id in self.tracks:
            track = self.tracks[self.selected_id]
            x, y, w, h = track['bbox']
            
            # 绘制边界框
            cv2.rectangle(result_frame, (int(x), int(y)), 
                         (int(x + w), int(y + h)), color, 3)
            
            # 绘制轨迹
            if 'trajectory' in track and len(track['trajectory']) > 1:
                traj = np.array(track['trajectory'], dtype=np.int32)
                for i in range(1, len(traj)):
                    cv2.line(result_frame, tuple(traj[i-1]), tuple(traj[i]), 
                            color, 2)
            
            # 绘制中文标签
            label_text = self.label
            
            # 确保标签位置在图像内
            label_y = max(25, y - 5)
            
            result_frame = draw_text_on_image(
                result_frame, 
                label_text, 
                position=(x, label_y - 20),
                font_size=20,
                color=(255, 255, 255), 
                bg_color=color
            )
        
        return result_frame
    
    def _match_detections(self, detections):
        """匹配检测结果与现有跟踪记录
        
        Args:
            detections: YOLO检测器返回的检测结果列表
        """
        if not detections:
            # 如果没有检测到任何目标，将所有跟踪记录标记为丢失
            for track_id in self.tracks:
                self.tracks[track_id]['missed'] += 1
            return
        
        # 计算所有检测结果和跟踪记录之间的IOU
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, (track_id, track) in enumerate(self.tracks.items()):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track['bbox'], detection['bbox'])
        
        # 贪婪匹配：为每个跟踪记录找到最佳匹配
        track_indices = list(range(len(self.tracks)))
        detection_indices = list(range(len(detections)))
        
        matched_tracks = {}
        
        # 按IOU降序排序
        matches = []
        for i in track_indices:
            for j in detection_indices:
                if iou_matrix[i, j] > self.iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # 贪婪匹配
        track_used = set()
        detection_used = set()
        
        for i, j, iou in matches:
            if i in track_used or j in detection_used:
                continue
                
            track_used.add(i)
            detection_used.add(j)
            
            # 获取对应的track_id
            track_id = list(self.tracks.keys())[i]
            detection = detections[j]
            
            # 更新跟踪记录
            self.tracks[track_id]['bbox'] = detection['bbox']
            self.tracks[track_id]['missed'] = 0  # 重置丢失计数
            self.tracks[track_id]['last_detection'] = detection['bbox']
            
            # 更新标签（如果是选中的目标）
            if track_id == self.selected_id:
                self.label = detection['chinese_name']
                self.tracks[track_id]['label'] = detection['chinese_name']
            
            matched_tracks[track_id] = self.tracks[track_id]
        
        # 处理未匹配的跟踪记录
        for i, (track_id, track) in enumerate(self.tracks.items()):
            if i not in track_used:
                track['missed'] += 1
                matched_tracks[track_id] = track
        
        # 处理未匹配的检测结果（创建新的跟踪记录）
        for j in detection_indices:
            if j not in detection_used:
                detection = detections[j]
                
                # 忽略置信度低的检测结果
                if detection['confidence'] < self.confidence_threshold:
                    continue
                    
                # 创建新的跟踪记录
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': detection['bbox'],
                    'label': detection['chinese_name'],
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'age': 0,
                    'missed': 0,
                    'trajectory': deque(maxlen=self.track_history_len),
                    'last_detection': detection['bbox']
                }
                
                # 添加当前位置到轨迹
                x, y, w, h = detection['bbox']
                center_x = x + w // 2
                center_y = y + h // 2
                self.tracks[track_id]['trajectory'].append((center_x, center_y))
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IOU
        
        Args:
            box1, box2: 边界框 (x, y, w, h)
            
        Returns:
            IOU值
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算各个边界
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # 检查是否有交集
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # 计算交集面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算两个边界框的面积
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # 计算并集面积
        union_area = box1_area + box2_area - intersection_area
        
        # 计算IOU
        return intersection_area / union_area if union_area > 0 else 0.0


class SmartTracker:
    """带有智能标签识别功能的跟踪器"""
    def __init__(self, api_client=None):
        self.template = None
        self.bbox = None
        self.prev_gray = None
        self.points = []
        self.search_margin = 20  # 搜索区域边界
        self.label = "目标"  # 默认中文标签
        self.confidence = 0.0  # 标签置信度
        self.api_client = api_client  # 大模型API客户端
        self.recognition_cooldown = 5  # 识别冷却时间（增加到5秒以降低API压力）
        self.last_recognition_time = 0  # 上次识别时间
        self.recognition_attempts = 0  # 识别尝试次数
        self.max_recognition_attempts = 3  # 最大尝试次数
        self.is_recognizing = False  # 识别状态标志，避免并发调用
    
    def init(self, frame, bbox, label=None):
        """初始化跟踪器
        
        Args:
            frame: 视频帧
            bbox: 边界框 (x, y, w, h)
            label: 目标标签，如果为None则尝试自动识别
        """
        x, y, w, h = [int(v) for v in bbox]
        self.bbox = (x, y, w, h)
        
        # 如果提供了标签，直接使用
        if label:
            self.label = label
        
        # 边界检查
        if y+h > frame.shape[0] or x+w > frame.shape[1]:
            h = min(h, frame.shape[0]-y)
            w = min(w, frame.shape[1]-x)
            self.bbox = (x, y, w, h)
            
        if h <= 0 or w <= 0:
            return False
            
        # 存储模板
        self.template = frame[y:y+h, x:x+w].copy()
        
        # 转换为灰度图用于跟踪
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            
        self.prev_gray = gray
        
        # 初始化跟踪点 - 使用边界框内的角点
        roi = gray[y:y+h, x:x+w]
        self.points = cv2.goodFeaturesToTrack(roi, 10, 0.01, 10)
        
        if self.points is not None:
            self.points = self.points.reshape(-1, 2)
            # 转换为绝对坐标
            for i in range(len(self.points)):
                self.points[i][0] += x
                self.points[i][1] += y
        else:
            # 如果没有找到角点，使用中心点
            self.points = np.array([[x + w/2, y + h/2]])
        
        # 重置识别状态
        self.recognition_attempts = 0
        self.last_recognition_time = 0
        
        # 如果没有提供标签且有API客户端，尝试自动识别
        if not label and self.api_client:
            self.recognize_object(frame)
        
        return True
    
    def recognize_object(self, frame):
        """使用大模型API识别目标物体
        
        Args:
            frame: 视频帧
        """
        # 检查是否已经在进行识别
        if self.is_recognizing:
            return
            
        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_recognition_time < self.recognition_cooldown:
            return
            
        # 检查最大尝试次数
        if self.recognition_attempts >= self.max_recognition_attempts and self.label != "目标":
            return
            
        # 检查API客户端
        if not self.api_client:
            logger.warning("未提供API客户端，无法进行物体识别")
            return
        
        try:
            self.is_recognizing = True  # 设置识别状态标志
            self.last_recognition_time = current_time  # 更新识别时间
            self.recognition_attempts += 1  # 增加尝试次数
            
            # 获取目标区域
            x, y, w, h = self.bbox
            
            # 边界检查
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                self.is_recognizing = False
                return
            
            # 裁剪目标区域，并包括一些上下文
            context_x = max(0, x - w//4)
            context_y = max(0, y - h//4)
            context_w = min(frame.shape[1] - context_x, w + w//2)
            context_h = min(frame.shape[0] - context_y, h + h//2)
            
            object_region = frame[context_y:context_y+context_h, context_x:context_x+context_w]
            
            # 转换为PIL图像
            if len(object_region.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(object_region)
            
            # 调用API识别物体
            logger.info(f"请求API识别物体... (尝试次数: {self.recognition_attempts}/{self.max_recognition_attempts})")
            
            # 异步方式优化 - 为了示例简化，仍然使用同步调用
            label = self.api_client.analyze_image(pil_img)
            
            # 更新标签
            if label and label != "识别失败" and label != "API错误" and label != "未知目标":
                self.label = label.strip()
                logger.info(f"识别结果: {self.label}")
            else:
                logger.warning(f"物体识别失败: {label}")
                
        except Exception as e:
            logger.error(f"物体识别过程中出错: {e}")
        finally:
            self.is_recognizing = False  # 重置识别状态标志
    
    def update(self, frame):
        """更新跟踪器"""
        if self.bbox is None or self.prev_gray is None:
            return False, (0, 0, 0, 0)
            
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if len(self.points) < 1:
            return False, (0, 0, 0, 0)
            
        try:
            # 确保points是正确的形状: Nx1x2
            pts = self.points.reshape(-1, 1, 2).astype(np.float32)
            
            # 计算光流
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts, None)
                
            if new_points is None or len(new_points) == 0:
                return False, self.bbox
                
            # 确保status是一维数组
            status = status.reshape(-1)
            
            # 选择好的点 - 使用布尔掩码
            good_mask = status == 1
            
            if not np.any(good_mask):
                return False, self.bbox
                
            # 使用布尔掩码过滤点
            good_new = new_points[good_mask].reshape(-1, 2)
            good_old = pts[good_mask].reshape(-1, 2)
            
            if len(good_new) < 1:
                return False, self.bbox
                
            # 计算移动
            movement = good_new - good_old
            if len(movement) > 0:
                median_move = np.median(movement, axis=0)
            
                # 更新边界框
                x, y, w, h = self.bbox
                new_x = max(0, int(x + median_move[0]))
                new_y = max(0, int(y + median_move[1]))
                
                # 确保边界框在图像范围内
                frame_h, frame_w = frame.shape[:2] if len(frame.shape) > 2 else frame.shape
                new_x = min(new_x, frame_w - w)
                new_y = min(new_y, frame_h - h)
                
                # 更新状态
                self.bbox = (new_x, new_y, w, h)
                self.prev_gray = gray
                self.points = good_new
                
                # 周期性地尝试识别物体
                if self.api_client and self.label == "目标" and not self.is_recognizing and \
                   time.time() - self.last_recognition_time > self.recognition_cooldown and \
                   self.recognition_attempts < self.max_recognition_attempts:
                    # 使用线程池或后台线程优化，这里简化处理
                    self.recognize_object(frame)
                
                return True, self.bbox
            else:
                return False, self.bbox
                
        except Exception as e:
            logger.error(f"跟踪更新错误: {e}")
            return False, self.bbox
    
    def draw_bbox(self, frame, color=(0, 255, 0)):
        """在帧上绘制边界框和标签"""
        if self.bbox is None:
            return frame
            
        x, y, w, h = [int(v) for v in self.bbox]
        
        # 确保边界框在图像内
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return frame
        
        try:
            # 绘制边界框
            frame_with_box = cv2.rectangle(frame.copy(), (x, y), (x+w, y+h), color, 2)
            
            # 绘制中文标签
            label_text = f"{self.label}"
            if self.is_recognizing:
                label_text += " 识别中..."
                
            # 确保标签位置在图像内
            label_y = max(25, y - 5)
            
            frame_with_box = draw_text_on_image(
                frame_with_box,
                label_text,
                position=(x, label_y - 20),
                font_size=20,
                color=(255, 255, 255),
                bg_color=color
            )
            
            return frame_with_box
            
        except Exception as e:
            logger.error(f"绘制边界框错误: {e}")
            return frame


class SimpleTracker:
    """自定义基础跟踪器"""
    def __init__(self):
        self.template = None
        self.bbox = None
        self.prev_gray = None
        self.points = []
        self.search_margin = 20  # 搜索区域边界
        self.label = "目标"  # 默认中文标签
    
    def init(self, frame, bbox, label=None):
        """初始化跟踪器"""
        x, y, w, h = [int(v) for v in bbox]
        self.bbox = (x, y, w, h)
        
        if label:
            self.label = label
        
        # 边界检查
        if y+h > frame.shape[0] or x+w > frame.shape[1]:
            h = min(h, frame.shape[0]-y)
            w = min(w, frame.shape[1]-x)
            self.bbox = (x, y, w, h)
            
        if h <= 0 or w <= 0:
            return False
            
        # 存储模板
        self.template = frame[y:y+h, x:x+w].copy()
        
        # 转换为灰度图用于跟踪
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            
        self.prev_gray = gray
        
        # 初始化跟踪点 - 使用边界框内的角点
        roi = gray[y:y+h, x:x+w]
        self.points = cv2.goodFeaturesToTrack(roi, 10, 0.01, 10)
        
        if self.points is not None:
            self.points = self.points.reshape(-1, 2)
            # 转换为绝对坐标
            for i in range(len(self.points)):
                self.points[i][0] += x
                self.points[i][1] += y
        else:
            # 如果没有找到角点，使用中心点
            self.points = np.array([[x + w/2, y + h/2]])
        
        return True
        
    def update(self, frame):
        """更新跟踪器"""
        if self.bbox is None or self.prev_gray is None:
            return False, (0, 0, 0, 0)
            
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        if len(self.points) < 1:
            return False, (0, 0, 0, 0)
            
        try:
            # 确保points是正确的形状: Nx1x2
            pts = self.points.reshape(-1, 1, 2).astype(np.float32)
            
            # 计算光流
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts, None)
                
            if new_points is None or len(new_points) == 0:
                return False, self.bbox
                
            # 确保status是一维数组
            status = status.reshape(-1)
            
            # 选择好的点 - 使用布尔掩码
            good_mask = status == 1
            
            if not np.any(good_mask):
                return False, self.bbox
                
            # 使用布尔掩码过滤点
            good_new = new_points[good_mask].reshape(-1, 2)
            good_old = pts[good_mask].reshape(-1, 2)
            
            if len(good_new) < 1:
                return False, self.bbox
                
            # 计算移动
            movement = good_new - good_old
            if len(movement) > 0:
                median_move = np.median(movement, axis=0)
            
                # 更新边界框
                x, y, w, h = self.bbox
                new_x = max(0, int(x + median_move[0]))
                new_y = max(0, int(y + median_move[1]))
                
                # 确保边界框在图像范围内
                frame_h, frame_w = frame.shape[:2] if len(frame.shape) > 2 else frame.shape
                new_x = min(new_x, frame_w - w)
                new_y = min(new_y, frame_h - h)
                
                # 更新状态
                self.bbox = (new_x, new_y, w, h)
                self.prev_gray = gray
                self.points = good_new
                
                return True, self.bbox
            else:
                return False, self.bbox
                
        except Exception as e:
            logger.error(f"跟踪更新错误: {e}")
            return False, self.bbox
    
    def draw_bbox(self, frame, color=(0, 255, 0)):
        """在帧上绘制边界框和标签"""
        if self.bbox is None:
            return frame
            
        x, y, w, h = [int(v) for v in self.bbox]
        
        # 边界检查
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return frame
        
        # 绘制边界框
        frame_with_box = cv2.rectangle(frame.copy(), (x, y), (x+w, y+h), color, 2)
        
        # 绘制中文标签
        label_text = f"{self.label}"
        
        # 确保标签位置在图像内
        label_y = max(25, y - 5)
        
        frame_with_box = draw_text_on_image(
            frame_with_box, 
            label_text, 
            position=(x, label_y - 20),
            font_size=20,
            color=(255, 255, 255), 
            bg_color=color
        )
        
        return frame_with_box


class TemplateMatchingTracker:
    """模板匹配跟踪器"""
    def __init__(self):
        self.template = None
        self.bbox = None
        self.search_area = 40  # 搜索窗口大小
        self.label = "目标"  # 默认中文标签
        
    def init(self, frame, bbox, label=None):
        """初始化跟踪器"""
        x, y, w, h = [int(v) for v in bbox]
        self.bbox = (x, y, w, h)
        
        if label:
            self.label = label
        
        # 边界检查
        if y+h > frame.shape[0] or x+w > frame.shape[1]:
            h = min(h, frame.shape[0]-y)
            w = min(w, frame.shape[1]-x)
            self.bbox = (x, y, w, h)
            
        if h <= 0 or w <= 0:
            return False
            
        # 转换为灰度图用于模板匹配
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            
        # 存储模板
        self.template = gray[y:y+h, x:x+w].copy()
        return True
        
    def update(self, frame):
        """更新跟踪器"""
        if self.template is None or self.bbox is None:
            return False, (0, 0, 0, 0)
            
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            
        x, y, w, h = self.bbox
        
        # 定义搜索区域
        search_x = max(0, x - self.search_area)
        search_y = max(0, y - self.search_area)
        search_w = min(gray.shape[1] - search_x, w + 2 * self.search_area)
        search_h = min(gray.shape[0] - search_y, h + 2 * self.search_area)
        
        if search_w <= 0 or search_h <= 0:
            return False, self.bbox
            
        # 提取搜索区域
        search_region = gray[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # 执行模板匹配
        if search_region.shape[0] < self.template.shape[0] or search_region.shape[1] < self.template.shape[1]:
            return False, self.bbox
            
        try:
            result = cv2.matchTemplate(search_region, self.template, cv2.TM_CCOEFF_NORMED)
            
            # 找到最佳匹配位置
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # 更新边界框
            new_x = search_x + max_loc[0]
            new_y = search_y + max_loc[1]
            
            self.bbox = (new_x, new_y, w, h)
            return True, self.bbox
            
        except Exception as e:
            logger.error(f"模板匹配错误: {e}")
            return False, self.bbox
    
    def draw_bbox(self, frame, color=(0, 255, 0)):
        """在帧上绘制边界框和标签"""
        if self.bbox is None:
            return frame
            
        x, y, w, h = [int(v) for v in self.bbox]
        
        # 边界检查
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return frame
        
        # 绘制边界框
        frame_with_box = cv2.rectangle(frame.copy(), (x, y), (x+w, y+h), color, 2)
        
        # 绘制中文标签
        label_text = f"{self.label}"
        
        # 确保标签位置在图像内
        label_y = max(25, y - 5)
        
        frame_with_box = draw_text_on_image(
            frame_with_box, 
            label_text, 
            position=(x, label_y - 20),
            font_size=20,
            color=(255, 255, 255), 
            bg_color=color
        )
        
        return frame_with_box


class OpenCVTrackerWrapper:
    """包装OpenCV跟踪器，添加额外功能"""
    
    def __init__(self, tracker, api_client=None):
        """初始化包装器
        
        Args:
            tracker: OpenCV跟踪器对象
            api_client: 大模型API客户端
        """
        self.tracker = tracker
        self.label = "目标"  # 添加标签属性
        self.bbox = None
        self.api_client = api_client  # 大模型API客户端
        self.recognition_cooldown = 5  # 识别冷却时间
        self.last_recognition_time = 0  # 上次识别时间
        self.recognition_attempts = 0  # 识别尝试次数
        self.max_recognition_attempts = 3  # 最大尝试次数
        self.is_recognizing = False  # 识别状态标志
        self.orig_frame = None  # 保存原始帧用于识别
    
    def init(self, frame, bbox, label=None):
        """初始化跟踪器"""
        if label:
            self.label = label
        
        x, y, w, h = [int(v) for v in bbox]
        self.bbox = (x, y, w, h)
        
        # 保存原始帧
        self.orig_frame = frame.copy()
        
        # 重置识别状态
        self.recognition_attempts = 0
        self.last_recognition_time = 0
        
        # OpenCV跟踪器要求的边界框格式
        try:
            result = self.tracker.init(frame, (x, y, w, h))
            
            # 如果没有提供标签且有API客户端，尝试自动识别
            if not label and self.api_client:
                self.recognize_object(frame)
                
            return result
        except Exception as e:
            logger.error(f"初始化OpenCV跟踪器错误: {e}")
            return False
    
    def recognize_object(self, frame):
        """使用大模型API识别目标物体"""
        # 与SmartTracker中的实现相同
        # 检查是否已经在进行识别
        if self.is_recognizing:
            return
            
        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_recognition_time < self.recognition_cooldown:
            return
            
        # 检查最大尝试次数
        if self.recognition_attempts >= self.max_recognition_attempts and self.label != "目标":
            return
            
        # 检查API客户端
        if not self.api_client:
            return
        
        try:
            self.is_recognizing = True
            self.last_recognition_time = current_time
            self.recognition_attempts += 1
            
            # 获取目标区域
            x, y, w, h = self.bbox
            
            # 边界检查
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                self.is_recognizing = False
                return
                
            # 裁剪目标区域，并包括一些上下文
            context_x = max(0, x - w//4)
            context_y = max(0, y - h//4)
            context_w = min(frame.shape[1] - context_x, w + w//2)
            context_h = min(frame.shape[0] - context_y, h + h//2)
            
            object_region = frame[context_y:context_y+context_h, context_x:context_x+context_w]
            
            # 转换为PIL图像
            if len(object_region.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(object_region, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(object_region)
            
            # 调用API识别物体
            logger.info(f"请求API识别物体... (尝试次数: {self.recognition_attempts}/{self.max_recognition_attempts})")
            label = self.api_client.analyze_image(pil_img)
            
            # 更新标签
            if label and label != "识别失败" and label != "API错误" and label != "未知目标":
                self.label = label.strip()
                logger.info(f"识别结果: {self.label}")
            else:
                logger.warning(f"物体识别失败: {label}")
                
        except Exception as e:
            logger.error(f"物体识别过程中出错: {e}")
        finally:
            self.is_recognizing = False  # 重置识别状态标志
    
    def update(self, frame):
        """更新跟踪器"""
        try:
            success, box = self.tracker.update(frame)
            if success:
                self.bbox = box
                
                # 保存当前帧用于可能的识别
                self.orig_frame = frame.copy()
                
                # 周期性地尝试识别物体
                if self.api_client and self.label == "目标" and not self.is_recognizing and \
                   time.time() - self.last_recognition_time > self.recognition_cooldown and \
                   self.recognition_attempts < self.max_recognition_attempts:
                    self.recognize_object(frame)
            
            return success, box
        except Exception as e:
            logger.error(f"更新OpenCV跟踪器错误: {e}")
            return False, self.bbox if self.bbox else (0, 0, 0, 0)
    
    def draw_bbox(self, frame, color=(0, 255, 0)):
        """在帧上绘制边界框和标签"""
        if self.bbox is None:
            return frame
        
        try:
            x, y, w, h = [int(v) for v in self.bbox]
            
            # 边界检查
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return frame
            
            # 绘制边界框
            frame_with_box = cv2.rectangle(frame.copy(), (x, y), (x+w, y+h), color, 2)
            
            # 绘制中文标签
            label_text = f"{self.label}"
            if self.is_recognizing:
                label_text += " 识别中..."
                
            # 确保标签位置在图像内
            label_y = max(25, y - 5)
            
            frame_with_box = draw_text_on_image(
                frame_with_box, 
                label_text, 
                position=(x, label_y - 20),
                font_size=20,
                color=(255, 255, 255), 
                bg_color=color
            )
            
            return frame_with_box
        except Exception as e:
            logger.error(f"绘制边界框错误: {e}")
            return frame


# 定义全局跟踪器字典
TRACKING_ALGORITHMS = {}

# 创建全局API客户端对象
_global_api_client = None

# 全局YOLO检测器实例
_global_yolo_detector = None

def init_yolo_detector(model_name="yolov12s"):
    """初始化全局YOLO检测器
    
    Args:
        model_name: 模型名称
    
    Returns:
        YOLOv12检测器实例
    """
    global _global_yolo_detector
    
    if _global_yolo_detector is None:
        _global_yolo_detector = YOLOv12Detector(model_name=model_name)
        _global_yolo_detector.initialize()
        logger.info(f"全局YOLO检测器已初始化，使用模型: {model_name}")
    
    return _global_yolo_detector

def _initialize_tracking_algorithms(api_client=None, yolo_detector=None):
    """初始化跟踪算法字典
    
    Args:
        api_client: 可选的API客户端，用于智能识别
        yolo_detector: 可选的YOLO检测器，用于目标检测
    """
    global TRACKING_ALGORITHMS
    
    # 首先添加自定义跟踪器
    TRACKING_ALGORITHMS["基础"] = lambda: SimpleTracker() if api_client is None else SmartTracker(api_client=api_client)
    TRACKING_ALGORITHMS["模板匹配"] = lambda: TemplateMatchingTracker()
    TRACKING_ALGORITHMS["智能"] = lambda: SmartTracker(api_client=api_client)
    
    # 添加YOLO跟踪器
    if yolo_detector is not None:
        TRACKING_ALGORITHMS["YOLO"] = lambda: YOLOTracker(detector=yolo_detector, api_client=api_client)
    
    # 添加英文别名
    TRACKING_ALGORITHMS["Basic"] = lambda: SimpleTracker() if api_client is None else SmartTracker(api_client=api_client)
    TRACKING_ALGORITHMS["Template"] = lambda: TemplateMatchingTracker()
    TRACKING_ALGORITHMS["Smart"] = lambda: SmartTracker(api_client=api_client)
    
    logger.info("内置跟踪器已添加")
    
    # 添加 OpenCV 跟踪器，并使用包装类
    if hasattr(cv2, 'legacy'):
        for name in ["CSRT", "KCF", "MIL", "MOSSE", "MedianFlow", "TLD", "BOOSTING"]:
            tracker_create_fn = f'Tracker{name}_create'
            if hasattr(cv2.legacy, tracker_create_fn):
                # 使用内部函数来捕获 name 变量的实际值
                def create_fn(tracker_name=name):
                    try:
                        creator = getattr(cv2.legacy, f'Tracker{tracker_name}_create')
                        cv_tracker = creator()
                        # 使用包装类包装OpenCV跟踪器，添加智能标签支持
                        return OpenCVTrackerWrapper(cv_tracker, api_client=api_client)
                    except Exception as e:
                        logger.error(f"创建 {tracker_name} 跟踪器时出错: {e}")
                        return None
                
                TRACKING_ALGORITHMS[name] = create_fn
                logger.info(f"添加了 OpenCV legacy {name} 跟踪器")
    else:
        # 尝试直接从 cv2 创建
        for name in ["CSRT", "KCF", "MIL", "MOSSE", "MedianFlow", "TLD", "BOOSTING"]:
            tracker_create_fn = f'Tracker{name}_create'
            if hasattr(cv2, tracker_create_fn):
                # 使用内部函数来捕获 name 变量的实际值
                def create_fn(tracker_name=name):
                    try:
                        creator = getattr(cv2, f'Tracker{tracker_name}_create')
                        cv_tracker = creator()
                        # 使用包装类包装OpenCV跟踪器，添加智能标签支持
                        return OpenCVTrackerWrapper(cv_tracker, api_client=api_client)
                    except Exception as e:
                        logger.error(f"创建 {tracker_name} 跟踪器时出错: {e}")
                        return None
                
                TRACKING_ALGORITHMS[name] = create_fn
                logger.info(f"添加了 OpenCV {name} 跟踪器")


def init_api_client(api_key=None, model_name="qwen-vl-max", init_yolo=True, yolo_model="yolov12s"):
    """初始化全局API客户端
    
    Args:
        api_key: API密钥
        model_name: 模型名称
        init_yolo: 是否同时初始化YOLO检测器
        yolo_model: YOLO模型名称
    
    Returns:
        API客户端实例
    """
    global _global_api_client
    
    if _global_api_client is None:
        _global_api_client = ModelAPIClient(api_key=api_key, model_name=model_name)
        
        # 如果需要，同时初始化YOLO检测器
        yolo_detector = None
        if init_yolo:
            try:
                yolo_detector = init_yolo_detector(model_name=yolo_model)
            except Exception as e:
                logger.error(f"初始化YOLO检测器时出错: {e}")
        
        # 初始化跟踪算法
        _initialize_tracking_algorithms(_global_api_client, yolo_detector)
        
        logger.info(f"全局API客户端已初始化，使用模型: {model_name}")
    
    return _global_api_client

def create_tracker(algorithm="基础", label="目标", api_client=None, yolo_model=None):
    """创建跟踪器
    
    Args:
        algorithm: 跟踪算法名称，默认为"基础"
        label: 目标标签，默认为"目标"
        api_client: API客户端，为None则使用全局客户端
        yolo_model: YOLO模型名称，指定时将优先使用该模型
        
    Returns:
        跟踪器实例
    """
    global TRACKING_ALGORITHMS, _global_api_client, _global_yolo_detector
    
    # 如果尚未初始化跟踪算法
    if not TRACKING_ALGORITHMS:
        # 使用提供的API客户端或全局客户端
        client = api_client or _global_api_client
        
        # 检查是否有YOLO环境变量
        yolo_available = os.environ.get("YOLO_AVAILABLE", "0") == "1"
        yolo_detector = None
        
        # 如果YOLO可用，初始化YOLO检测器
        if yolo_available:
            try:
                # 使用指定模型或默认模型
                model_name = yolo_model or "yolov12s"
                yolo_detector = init_yolo_detector(model_name)
                
                if yolo_detector and not yolo_detector.initialized:
                    logger.warning(f"YOLO检测器初始化失败，将不使用YOLO相关功能")
                    yolo_detector = None
                    
            except Exception as e:
                logger.error(f"初始化YOLO检测器时出错: {e}")
                yolo_detector = None
        
        # 初始化跟踪算法
        _initialize_tracking_algorithms(client, yolo_detector)
    
    # 特殊处理YOLO算法
    if algorithm.upper() == "YOLO":
        yolo_available = os.environ.get("YOLO_AVAILABLE", "0") == "1"
        
        if not yolo_available:
            logger.warning("请求YOLO跟踪器但YOLO未启用，将使用智能跟踪器代替")
            algorithm = "智能" if "智能" in TRACKING_ALGORITHMS else "基础"
        elif "YOLO" not in TRACKING_ALGORITHMS:
            # 尝试动态添加YOLO跟踪器
            try:
                # 使用指定模型或默认模型
                model_name = yolo_model or "yolov12s"
                yolo_detector = init_yolo_detector(model_name)
                
                if yolo_detector and yolo_detector.initialized:
                    TRACKING_ALGORITHMS["YOLO"] = lambda: YOLOTracker(detector=yolo_detector, api_client=_global_api_client)
                    logger.info(f"动态添加YOLO跟踪器成功")
                else:
                    logger.warning(f"YOLO检测器初始化失败，将不使用YOLO跟踪器")
                    algorithm = "智能" if "智能" in TRACKING_ALGORITHMS else "基础"
                    
            except Exception as e:
                logger.error(f"动态添加YOLO跟踪器时出错: {e}")
                algorithm = "智能" if "智能" in TRACKING_ALGORITHMS else "基础"
    
    # 最终检查算法是否可用
    if algorithm not in TRACKING_ALGORITHMS:
        logger.warning(f"跟踪器 {algorithm} 不可用，使用基础跟踪器")
        algorithm = "基础"
        
    logger.info(f"创建跟踪器: {algorithm}, 标签: {label}")
    
    try:
        tracker = TRACKING_ALGORITHMS[algorithm]()
        if tracker is not None:
            # 设置标签
            tracker.label = label
            return tracker
    except Exception as e:
        logger.error(f"无法创建 {algorithm} 跟踪器: {e}")
    
    # 如果创建失败，使用基础跟踪器
    logger.warning(f"回退到基础跟踪器")
    
    # 尝试创建智能跟踪器，如果失败则使用基础跟踪器
    try:
        if _global_api_client:
            tracker = SmartTracker(api_client=_global_api_client)
        else:
            tracker = SimpleTracker()
        tracker.label = label
        return tracker
    except Exception as e:
        logger.error(f"创建回退跟踪器时出错: {e}，将使用最基础的跟踪器")
        tracker = SimpleTracker()
        tracker.label = label
        return tracker

def get_available_algorithms():
    """获取可用的跟踪算法列表"""
    # 如果尚未初始化跟踪算法
    if not TRACKING_ALGORITHMS:
        # 检查是否有YOLO环境变量
        yolo_available = os.environ.get("YOLO_AVAILABLE", "0") == "1"
        yolo_detector = None
        
        # 如果YOLO可用，初始化YOLO检测器
        if yolo_available:
            try:
                yolo_detector = init_yolo_detector()
            except Exception as e:
                logger.error(f"初始化YOLO检测器时出错: {e}")
        
        # 初始化跟踪算法
        _initialize_tracking_algorithms(_global_api_client, yolo_detector)
        
    # 确保基础跟踪器总是可用
    available = []
    for name in TRACKING_ALGORITHMS.keys():
        try:
            # 尝试创建一个跟踪器实例，检查是否能正常工作
            tracker = create_tracker(name)
            if tracker is not None:
                available.append(name)
        except Exception as e:
            logger.warning(f"检查 {name} 跟踪器时出错: {e}")
    
    logger.info(f"可用跟踪算法: {available}")
    return available

def track_and_draw(tracker, frame, bbox=None, label=None):
    """跟踪目标并在图像上绘制结果"""
    if bbox is not None:
        # 初始化跟踪器
        success = tracker.init(frame, bbox, label)
        if not success:
            return False, None, frame
    else:
        # 更新跟踪
        success, bbox = tracker.update(frame)
        if not success:
            return False, None, frame
    
    # 绘制结果
    if hasattr(tracker, "draw_bbox"):
        # 使用跟踪器自身的绘制方法
        frame_with_box = tracker.draw_bbox(frame)
    else:
        # 通用绘制方法
        x, y, w, h = [int(v) for v in bbox]
        frame_with_box = cv2.rectangle(frame.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 绘制中文标签
        label_text = f"{getattr(tracker, 'label', '目标')}"
        frame_with_box = draw_text_on_image(
            frame_with_box,
            label_text,
            position=(x, y - 25),
            font_size=20,
            color=(255, 255, 255),
            bg_color=(0, 255, 0)
        )
    
    return success, bbox, frame_with_box

def create_yolo_tracker(api_key=None, model_name="qwen-vl-max", yolo_model="yolov12s"):
    """创建YOLO跟踪器
    
    Args:
        api_key: API密钥，如果为None则尝试从环境变量获取
        model_name: 大模型名称
        yolo_model: YOLO模型名称
        
    Returns:
        YOLO跟踪器实例
    """
    # 初始化或获取API客户端
    api_client = init_api_client(api_key=api_key, model_name=model_name, init_yolo=False)
    
    # 初始化或获取YOLO检测器
    yolo_detector = init_yolo_detector(model_name=yolo_model)
    
    # 创建YOLO跟踪器
    return YOLOTracker(detector=yolo_detector, api_client=api_client)

# 测试函数
def test_chinese_font():
    """测试中文字体是否正常工作"""
    # 创建一个空白图像
    img = np.ones((300, 500, 3), dtype=np.uint8) * 255
    
    try:
        # 绘制中文文本
        img = draw_text_on_image(
            img, 
            "中文测试 - 智能视频跟踪器", 
            position=(50, 50),
            font_size=30,
            color=(0, 0, 255),
            bg_color=None,
            outline=True
        )
        
        # 保存图像到文件 (可选，用于调试)
        cv2.imwrite("chinese_font_test.jpg", img)
        
        # 显示图像
        cv2.imshow("中文字体测试", img)
        cv2.waitKey(1000)  # 显示1秒
        cv2.destroyAllWindows()
        
        return True
    except Exception as e:
        logger.error(f"测试中文字体时出错: {e}")
        return False

def create_smart_tracker(api_key=None, model_name="qwen-vl-max"):
    """创建智能跟踪器
    
    Args:
        api_key: API密钥，如果为None则尝试从环境变量获取
        model_name: 模型名称
        
    Returns:
        智能跟踪器实例
    """
    # 初始化或获取API客户端
    api_client = init_api_client(api_key=api_key, model_name=model_name)
    
    # 创建智能跟踪器
    return SmartTracker(api_client=api_client)


if __name__ == "__main__":
    # 测试中文字体
    test_chinese_font()
    
    # 测试跟踪器
    print(f"可用跟踪算法: {get_available_algorithms()}")
    
    # 检查YOLO是否可用
    yolo_available = os.environ.get("YOLO_AVAILABLE", "0") == "1"
    if yolo_available:
        try:
            # 测试YOLO检测器
            detector = init_yolo_detector()
            if detector.initialized:
                print("YOLO检测器初始化成功")
                
                # 创建并测试YOLO跟踪器
                yolo_tracker = create_yolo_tracker()
                print(f"YOLO跟踪器创建成功: {yolo_tracker}")
        except Exception as e:
            print(f"YOLO测试失败: {e}")
    else:
        print("YOLO未启用，使用普通跟踪器")
        
    # 创建智能跟踪器（如果环境变量中有API密钥）
    try:
        tracker = create_smart_tracker()
        print(f"创建了智能跟踪器: {tracker}")
    except Exception as e:
        print(f"创建智能跟踪器失败: {e}，使用基础跟踪器")
        tracker = SimpleTracker()