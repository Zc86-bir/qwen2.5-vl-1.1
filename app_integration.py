import tkinter as tk
from video_analysis_window import VideoAnalysisWindow
from model_api import ModelAPIClient
import logging
import os

logger = logging.getLogger(__name__)

def add_video_analysis_to_menu(root, menu_bar):
    """将视频分析功能添加到菜单栏
    
    Parameters:
        root: 主窗口
        menu_bar: 菜单栏对象
    """
    # 创建API客户端
    api_client = ModelAPIClient(api_key=os.getenv("DASHSCOPE_API_KEY"))
    
    # 获取或创建"工具"菜单
    tools_menu = None
    for i in range(menu_bar.index("end") + 1):
        try:
            label = menu_bar.entrycget(i, "label")
            if label == "工具" or label == "Tools":
                tools_menu = menu_bar.winfo_children()[i]
                break
        except:
            continue
    
    if not tools_menu:
        # 如果没有工具菜单，创建一个
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="工具", menu=tools_menu)
    
    # 添加视频分析菜单项
    def open_video_analysis():
        # 隐藏当前界面
        current_frames = [f for f in root.winfo_children() if isinstance(f, tk.Frame)]
        for frame in current_frames:
            frame.pack_forget()
        
        # 创建视频分析窗口
        def return_callback():
            # 恢复主界面
            for frame in current_frames:
                frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建分析窗口
        VideoAnalysisWindow(
            root, 
            return_callback=return_callback,
            api_caller=api_client.analyze_video_frames
        )
    
    tools_menu.add_command(label="视频内容分析", command=open_video_analysis)
    logger.info("已添加视频内容分析功能到菜单")

def add_video_analysis_button(frame, button_container=None):
    """添加视频分析按钮到界面
    
    Parameters:
        frame: 主框架
        button_container: 按钮容器（如果为None，将在frame中创建）
    """
    import tkinter as tk
    from tkinter import ttk
    
    # 创建API客户端
    api_client = ModelAPIClient(api_key=os.getenv("DASHSCOPE_API_KEY"))
    
    # 如果没有提供按钮容器，创建一个
    if button_container is None:
        button_container = ttk.Frame(frame)
        button_container.pack(fill=tk.X, padx=10, pady=5)
    
    # 创建按钮
    def open_video_analysis():
        # 隐藏当前界面
        current_frames = [f for f in frame.winfo_children() if isinstance(f, tk.Frame)]
        for f in current_frames:
            f.pack_forget()
        
        # 创建视频分析窗口
        def return_callback():
            # 恢复主界面
            for f in current_frames:
                f.pack(fill=tk.BOTH, expand=True)
        
        # 创建分析窗口
        VideoAnalysisWindow(
            frame, 
            return_callback=return_callback,
            api_caller=api_client.analyze_video_frames
        )
    
    video_analysis_button = ttk.Button(
        button_container,
        text="视频内容分析",
        command=open_video_analysis
    )
    video_analysis_button.pack(side=tk.LEFT, padx=5)
    
    logger.info("已添加视频内容分析按钮到界面")