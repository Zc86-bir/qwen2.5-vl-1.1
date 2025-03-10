#!/usr/bin/env python3
import os
import sys
import subprocess
import pkg_resources
import shutil
import urllib.request
import zipfile
from pathlib import Path

# Required packages - 添加了ultralytics用于YOLOv12支持
required_packages = [
    'openai>=1.0.0', 
    'pillow>=9.0.0', 
    'numpy>=1.20.0',
    'tenacity>=8.0.0',
    'requests>=2.25.0',
    'ultralytics>=8.0.0'  # YOLOv12依赖
]

# YOLOv12模型配置
YOLO_MODELS = {
    "yolov12n.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12n.pt",
    "yolov12s.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12s.pt",
    "yolov12m.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12m.pt",
    "yolov12l.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12l.pt",
    "yolov12x.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt"
}

DEFAULT_YOLO_MODEL = "yolov12s.pt"  # 默认使用small模型平衡性能和速度

def check_packages():
    """Check if required packages are installed and install if missing"""
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = []
    
    for package in required_packages:
        package_name = package.split('>=')[0]
        if package_name not in installed:
            missing.append(package)
    
    if missing:
        print(f"安装缺失的依赖包: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("所有依赖包安装成功!")
    else:
        print("所有依赖包已安装.")

def check_opencv():
    """Check OpenCV version and available modules"""
    import cv2
    print(f"OpenCV 版本: {cv2.__version__}")
    
    # Check if legacy module is available
    has_legacy = hasattr(cv2, 'legacy')
    print(f"OpenCV legacy 模块可用: {has_legacy}")
    
    # 检测可用跟踪器
    trackers = []
    
    # Check for Basic and Template (custom trackers)
    trackers.append("Basic")
    trackers.append("Template")
    
    print(f"可用跟踪器: {', '.join(trackers)}")
    
    # 使用 tracking_utils 检测可用跟踪器
    try:
        from tracking_utils import get_available_algorithms
        available = get_available_algorithms()
        print(f"tracking_utils 可用跟踪器: {', '.join(available)}")
    except Exception as e:
        print(f"检查跟踪器时出错: {e}")

def setup_yolo_models():
    """设置YOLO模型，检查是否存在，如不存在则下载"""
    # 创建models目录
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 检查默认模型是否存在，如不存在则下载
    default_model_path = models_dir / DEFAULT_YOLO_MODEL
    if not default_model_path.exists():
        print(f"未找到默认YOLOv12模型: {DEFAULT_YOLO_MODEL}")
        download_model = input(f"是否下载YOLOv12模型? (y/n): ").lower() == 'y'
        
        if download_model:
            download_yolo_model(DEFAULT_YOLO_MODEL)
        else:
            print(f"警告: 未下载YOLOv12模型，YOLO跟踪器将不可用")
    else:
        print(f"找到YOLOv12模型: {default_model_path}")
        
    # 检查本地其他YOLO模型
    available_models = []
    for model_name in YOLO_MODELS.keys():
        if (models_dir / model_name).exists():
            available_models.append(model_name)
    
    if available_models:
        print(f"可用的YOLOv12模型: {', '.join(available_models)}")
    
    return default_model_path.exists()

def download_yolo_model(model_name):
    """下载YOLO模型"""
    models_dir = Path("models")
    model_path = models_dir / model_name
    
    if model_name not in YOLO_MODELS:
        print(f"错误: 未知的模型 {model_name}")
        return False
    
    url = YOLO_MODELS[model_name]
    print(f"正在下载 {model_name} 从 {url}...")
    
    try:
        # 显示下载进度的回调函数
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            sys.stdout.write(f"\r下载进度: {percent}% [{downloaded} / {total_size} 字节]")
            sys.stdout.flush()
            
        # 下载模型
        urllib.request.urlretrieve(url, model_path, reporthook=report_progress)
        print(f"\n{model_name} 下载成功!")
        return True
    except Exception as e:
        print(f"\n下载 {model_name} 时出错: {e}")
        return False

def check_yolo():
    """检查YOLO相关配置"""
    try:
        # 检查ultralytics是否安装
        import importlib
        ultralytics_spec = importlib.util.find_spec("ultralytics")
        if ultralytics_spec is None:
            print("未检测到ultralytics包，将安装...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
            print("ultralytics已安装")
        else:
            import ultralytics
            print(f"已安装ultralytics版本: {ultralytics.__version__}")
        
        # 检查并设置YOLO模型
        has_model = setup_yolo_models()
        
        # 检查GPU支持
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if gpu_available else 0
            device_name = torch.cuda.get_device_name(0) if gpu_available and device_count > 0 else "CPU"
            
            print(f"YOLO将运行在: {'GPU - ' + device_name if gpu_available else 'CPU'}")
            print(f"CUDA可用: {gpu_available}, GPU数量: {device_count}")
        except Exception as e:
            print(f"检查GPU支持时出错: {e}")
            print("YOLO将在CPU模式下运行")
        
        return has_model
    except Exception as e:
        print(f"检查YOLO配置时出错: {e}")
        return False

def main():
    """Main entry point for the application"""
    print("---------- 多模态分析系统初始化 ----------")
    
    # Check for Python version
    if sys.version_info < (3, 8):
        print("错误: 本应用需要Python 3.8或更高版本.")
        sys.exit(1)
        
    # 检查并安装依赖包
    print("\n[1/4] 检查依赖包...")
    check_packages()
    
    # 检测 OpenCV 是否已安装
    print("\n[2/4] 检查OpenCV...")
    try:
        import cv2
        print(f"检测到OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("未安装OpenCV，正在安装...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-contrib-python==4.8.1.78'])
    
    # 检查OpenCV安装
    check_opencv()
    
    # 检查YOLO配置
    print("\n[3/4] 检查YOLOv12配置...")
    has_yolo = check_yolo()
    
    # 检查API密钥
    print("\n[4/4] 检查API密钥...")
    if not os.getenv("DASHSCOPE_API_KEY"):
        api_key = input("请输入DashScope API密钥: ")
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
            print("API密钥已设置")
        else:
            print("警告: 未提供API密钥，应用程序可能无法正常工作.")
    else:
        print("检测到API密钥")
    
    # 设置YOLO环境变量，使主应用可以检测YOLO是否可用
    if has_yolo:
        os.environ["YOLO_AVAILABLE"] = "1"
    else:
        os.environ["YOLO_AVAILABLE"] = "0"
    
    # 启动应用
    print("\n---------- 启动多模态分析系统 ----------")
    try:
        print("正在启动应用...")
        import multimodal_app
        multimodal_app.main()
    except ImportError as e:
        print(f"导入应用模块时出错: {str(e)}")
        print("请确保所有Python文件都在同一目录下.")
        sys.exit(1)
    except Exception as e:
        print(f"启动应用时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()