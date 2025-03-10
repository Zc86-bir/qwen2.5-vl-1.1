import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import cv2
import threading
import time
from PIL import Image, ImageTk
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class VideoAnalysisWindow:
    def __init__(self, root, return_callback=None, api_caller=None):
        """初始化视频分析窗口
        
        Parameters:
            root: 主窗口
            return_callback: 返回主界面的回调函数
            api_caller: 调用大模型API的函数
        """
        self.root = root
        self.return_callback = return_callback
        self.api_caller = api_caller
        
        # 创建视频分析窗口的框架
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # 设置标题
        self.title_label = ttk.Label(
            self.frame, 
            text="视频内容分析", 
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=10)
        
        # 创建主内容区域
        self.content_frame = ttk.Frame(self.frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        
        # 左侧区域 - 视频选择和预览
        self.left_frame = ttk.Frame(self.content_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 视频选择区域
        self.file_frame = ttk.Frame(self.left_frame)
        self.file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path = tk.StringVar()
        self.file_label = ttk.Label(self.file_frame, text="视频文件:")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path, width=40)
        self.file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_button = ttk.Button(
            self.file_frame, 
            text="浏览...", 
            command=self.browse_file
        )
        self.browse_button.pack(side=tk.LEFT, padx=5)
        
        # 视频信息区域
        self.info_frame = ttk.LabelFrame(self.left_frame, text="视频信息")
        self.info_frame.pack(fill=tk.X, pady=5)
        
        self.video_info = tk.Text(self.info_frame, height=4, width=40, wrap=tk.WORD)
        self.video_info.pack(padx=5, pady=5, fill=tk.X, expand=True)
        self.video_info.insert(tk.END, "请选择视频文件...")
        self.video_info.config(state=tk.DISABLED)
        
        # 预览区域
        self.preview_frame = ttk.LabelFrame(self.left_frame, text="视频预览")
        self.preview_frame.pack(fill=tk.BOTH, pady=5, expand=True)
        
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="black", width=320, height=240)
        self.preview_canvas.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        # 右侧区域 - 分析结果
        self.right_frame = ttk.Frame(self.content_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 分析参数设置
        self.params_frame = ttk.LabelFrame(self.right_frame, text="分析参数")
        self.params_frame.pack(fill=tk.X, pady=5)
        
        self.max_frames_label = ttk.Label(self.params_frame, text="最大帧数:")
        self.max_frames_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.max_frames = tk.IntVar(value=8)
        self.max_frames_spinbox = ttk.Spinbox(
            self.params_frame, 
            from_=1, 
            to=20, 
            width=5,
            textvariable=self.max_frames
        )
        self.max_frames_spinbox.grid(row=0, column=1, padx=5, pady=5)
        
        self.interval_label = ttk.Label(self.params_frame, text="抽帧间隔(秒):")
        self.interval_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        self.interval = tk.DoubleVar(value=1.0)
        self.interval_spinbox = ttk.Spinbox(
            self.params_frame, 
            from_=0.1, 
            to=10.0, 
            increment=0.1,
            width=5,
            textvariable=self.interval
        )
        self.interval_spinbox.grid(row=0, column=3, padx=5, pady=5)
        
        # 提示标签
        self.query_label = ttk.Label(self.params_frame, text="分析提示词:")
        self.query_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.query_text = tk.Text(self.params_frame, height=2, width=40, wrap=tk.WORD)
        self.query_text.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky=tk.W+tk.E)
        self.query_text.insert(tk.END, "请分析这段视频内容，描述主要活动和重要细节。")
        
        # 分析结果区域
        self.results_frame = ttk.LabelFrame(self.right_frame, text="分析结果")
        self.results_frame.pack(fill=tk.BOTH, pady=5, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(
            self.results_frame, 
            wrap=tk.WORD,
            width=40
        )
        self.results_text.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        
        # 底部控制区域
        self.control_frame = ttk.Frame(self.frame)
        self.control_frame.pack(fill=tk.X, pady=10, padx=15)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            self.control_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # 按钮区域
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        self.analyze_button = ttk.Button(
            self.button_frame, 
            text="开始分析", 
            command=self.analyze_video
        )
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        self.return_button = ttk.Button(
            self.button_frame, 
            text="返回主界面", 
            command=self.return_to_main
        )
        self.return_button.pack(side=tk.RIGHT, padx=5)
        
        # 状态变量
        self.is_analyzing = False
        self.video_path = None
        self.cap = None
        self.preview_thread = None
        self.stop_preview = False
        
    def browse_file(self):
        """选择视频文件"""
        filetypes = (
            ("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv"),
            ("所有文件", "*.*")
        )
        
        # 打开文件对话框
        filepath = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=filetypes
        )
        
        if not filepath:
            return
            
        self.video_path = filepath
        self.file_path.set(filepath)
        
        # 显示视频信息
        self.display_video_info()
        
        # 开始预览
        self.start_preview()
    
    def display_video_info(self):
        """显示视频文件的信息"""
        if not self.video_path or not os.path.exists(self.video_path):
            return
            
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                messagebox.showerror("错误", "无法打开视频文件")
                return
                
            # 获取视频信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # 更新信息显示
            self.video_info.config(state=tk.NORMAL)
            self.video_info.delete(1.0, tk.END)
            info_text = f"文件名: {os.path.basename(self.video_path)}\n"
            info_text += f"尺寸: {width}x{height}\n"
            info_text += f"帧率: {fps:.2f} FPS\n"
            info_text += f"时长: {int(duration/60)}分{int(duration%60)}秒 ({frame_count} 帧)"
            self.video_info.insert(tk.END, info_text)
            self.video_info.config(state=tk.DISABLED)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"读取视频信息出错: {e}")
            messagebox.showerror("错误", f"无法读取视频信息: {e}")
    
    def start_preview(self):
        """开始预览视频"""
        # 停止之前的预览线程
        if self.preview_thread and self.preview_thread.is_alive():
            self.stop_preview = True
            self.preview_thread.join()
        
        self.stop_preview = False
        self.preview_thread = threading.Thread(target=self.preview_video)
        self.preview_thread.daemon = True
        self.preview_thread.start()
    
    def preview_video(self):
        """预览视频线程"""
        if not self.video_path:
            return
            
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_time = 1.0 / fps if fps > 0 else 0.1
            
            frame_count = 0
            while cap.isOpened() and not self.stop_preview and frame_count < 300:  # 限制帧数防止无限循环
                ret, frame = cap.read()
                if not ret:
                    # 循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # 动态获取画布尺寸以适应当前大小
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                # 确保画布尺寸有效（在初始化时可能为1）
                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width = 320  # 默认宽度
                    canvas_height = 240  # 默认高度
                
                frame = cv2.resize(frame, (canvas_width, canvas_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 转换为PhotoImage
                image = Image.fromarray(frame)
                
                # 使用异常处理确保不会因为窗口关闭而崩溃
                try:
                    photo = ImageTk.PhotoImage(image=image)
                    
                    # 在主线程中更新UI - 使用队列或事件而不是直接调用
                    if not self.stop_preview:
                        self.root.after_idle(self.update_preview, photo)
                except Exception as e:
                    logger.error(f"创建或更新图像时出错: {e}")
                    break
                
                # 控制播放速度
                time.sleep(frame_time * 3)  # 放慢播放速度
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            logger.error(f"预览视频出错: {e}", exc_info=True)
    
    def update_preview(self, photo):
        """更新预览画面"""
        try:
            if self.stop_preview:
                return
                
            # 检查canvas是否仍然存在
            if not self.preview_canvas.winfo_exists():
                logger.warning("Canvas已不存在，停止更新")
                self.stop_preview = True
                return
                
            # 保存对PhotoImage的引用
            self.current_photo = photo
            
            # 清除canvas并创建新图像
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        except Exception as e:
            logger.error(f"更新预览画面出错: {e}")
            self.stop_preview = True
    
    def analyze_video(self):
        """分析视频内容"""
        if self.is_analyzing:
            messagebox.showinfo("提示", "分析正在进行中，请等待...")
            return
            
        if not self.video_path:
            messagebox.showerror("错误", "请先选择视频文件")
            return
            
        if self.api_caller is None:
            messagebox.showerror("错误", "API调用接口未配置")
            return
        
        # 获取参数
        max_frames = self.max_frames.get()
        interval = self.interval.get()
        query = self.query_text.get("1.0", tk.END).strip()
        
        # 启动分析线程
        self.is_analyzing = True
        self.analyze_button.config(state=tk.DISABLED)
        self.status_var.set("正在分析...")
        self.progress_var.set(0)
        
        threading.Thread(
            target=self._analyze_process,
            args=(max_frames, interval, query),
            daemon=True
        ).start()
    
    def _analyze_process(self, max_frames, interval, query):
        """视频分析处理线程"""
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("错误", "无法打开视频文件"))
                return
            
            # 获取视频基本信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # 计算抽帧间隔
            frame_interval = int(interval * fps)
            if frame_interval < 1:
                frame_interval = 1
            
            # 确定抽取的帧
            frames = []
            frame_positions = []
            cur_pos = 0
            
            self.root.after(0, lambda: self.results_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.results_text.insert(tk.END, "正在抽取视频帧...\n"))
            
            while len(frames) < max_frames and cur_pos < frame_count:
                # 设置当前帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 调整图像尺寸
                frame = cv2.resize(frame, (480, 270))
                frames.append(frame)
                frame_positions.append(cur_pos)
                
                # 更新进度
                progress = min(100, (len(frames) / max_frames) * 50)  # 抽帧占总进度的50%
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda i=len(frames), m=max_frames: 
                               self.status_var.set(f"已抽取 {i}/{m} 帧"))
                
                # 更新下一帧位置
                cur_pos += frame_interval
            
            cap.release()
            
            # 将帧转换为base64编码
            self.root.after(0, lambda: self.results_text.insert(tk.END, "正在准备图像数据...\n"))
            
            # 构建发送给API的数据
            frames_data = []
            for i, frame in enumerate(frames):
                # 转换成jpg格式的二进制数据
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # 记录帧位置信息
                frame_pos = frame_positions[i]
                frame_time = frame_pos / fps if fps > 0 else 0
                minutes = int(frame_time // 60)
                seconds = int(frame_time % 60)
                
                frames_data.append({
                    'image': pil_img,
                    'position': f"{minutes:02d}:{seconds:02d}"
                })
            
            # 调用API进行分析
            self.root.after(0, lambda: self.results_text.insert(tk.END, "正在发送数据到AI模型进行分析...\n\n"))
            self.root.after(0, lambda: self.progress_var.set(60))  # 更新进度到60%
            
            # 构建API请求内容
            video_name = os.path.basename(self.video_path)
            video_duration = f"{int(duration//60):02d}:{int(duration%60):02d}"
            prompt = f"以下是来自视频 '{video_name}'（总时长 {video_duration}）的 {len(frames)} 个关键帧，"
            prompt += f"这些帧是按时间顺序排列的。{query}"
            
            # 调用API
            result = self.api_caller(prompt, frames_data)
            
            # 显示结果
            self.root.after(0, lambda: self.results_text.insert(tk.END, "AI 分析结果：\n\n"))
            self.root.after(0, lambda: self.results_text.insert(tk.END, result))
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.status_var.set("分析完成"))
            
        except Exception as e:
            logger.error(f"分析视频时出错: {e}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("错误", f"分析过程中出错: {e}"))
            self.root.after(0, lambda: self.status_var.set("分析失败"))
        finally:
            self.is_analyzing = False
            self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
    
    def return_to_main(self):
        """返回主界面"""
        # 停止预览
        self.stop_preview = True
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(0.5)
        
        # 销毁当前界面
        self.frame.destroy()
        
        # 调用回调
        if self.return_callback:
            self.return_callback()