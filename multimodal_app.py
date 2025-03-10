import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog 
from PIL import Image, ImageTk, ImageDraw
from io import BytesIO
import requests
import json
import math
import cv2
import numpy as np
import base64
import re
import uuid
from datetime import datetime
from pathlib import Path
import logging
from queue import Queue
import time
from urllib.parse import urlparse
import tracking_utils
from openai import OpenAI, APIError
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.visualization_utils import visualize_detection, visualize_tracking
from app_integration import add_video_analysis_to_menu, add_video_analysis_button
from tracking_utils import create_tracker, get_available_algorithms, track_and_draw





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

# Default configuration
DEFAULT_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "tracking_algorithm": "CSRT",
    "tracking_interval": 5,  # Process every 5th frame for tracking
    "visualization": {
        "show_labels": True,
        "show_scores": False,
        "color_scheme": "default"
    }
}

class CustomNotebook(ttk.Notebook):
    """Enhanced Notebook widget with tab management"""
    def __init__(self, *args, **kwargs):
        ttk.Notebook.__init__(self, *args, **kwargs)
        self._active = None
        self.bind("<ButtonPress-1>", self.on_tab_press)
        self.bind("<ButtonRelease-1>", self.on_tab_release)

    def on_tab_press(self, event):
        try:
            self._active = self.identify(event.x, event.y)
        except:
            self._active = None

    def on_tab_release(self, event):
        if self._active == "close":
            index = self.index("@%d,%d" % (event.x, event.y))
            self.forget(index)

class MediaPreviewCanvas(tk.Canvas):
    """Enhanced media preview component with zoom and pan"""
    def __init__(self, parent):
        super().__init__(parent, bg="#f8f9fa", highlightthickness=2)
        self.items = []
        self.scale = 1.0
        self.selected_item = None
        self.bind("<Configure>", self.on_resize)
        self.bind("<MouseWheel>", self.on_mousewheel)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<Button-1>", self.on_click)
        
    def on_resize(self, event):
        self.update_previews()

    def on_mousewheel(self, event):
        # Zoom in/out with mouse wheel
        if event.delta > 0:
            self.scale *= 1.1
        else:
            self.scale *= 0.9
        self.update_previews()

    def on_drag(self, event):
        # Pan the canvas
        self.scan_dragto(event.x, event.y, gain=1)

    def on_click(self, event):
        # Start drag operation and select item
        self.scan_mark(event.x, event.y)
        
        # Find which item was clicked
        x, y = self.canvasx(event.x), self.canvasy(event.y)
        for i, item in enumerate(self.items):
            # Get item position (approximate)
            items = self.find_withtag("item_" + str(i))
            if items:
                bbox = self.bbox(items[0])
                if bbox and (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                    self.select_item(i)
                    break

    def select_item(self, index):
        """Select an item and highlight it"""
        if not 0 <= index < len(self.items):
            return
            
        self.selected_item = index
        self.update_previews()  # Redraw with selection highlighted

    def update_previews(self):
        """Update the preview display with current items and selection"""
        self.delete("all")
        x_offset = 10
        y_offset = 10
        max_row_height = 0
        
        for i, item in enumerate(self.items):
            # Create item tag for selection
            item_tag = "item_" + str(i)
            
            # Check if this is the selected item
            is_selected = (i == self.selected_item)
            
            if item["type"] == "image":
                # Apply zoom scale to images
                img = item["preview"]
                
                # Draw selection border if selected
                if is_selected:
                    rect_id = self.create_rectangle(
                        x_offset - 3, y_offset - 3, 
                        x_offset + int(160 * self.scale) + 3, 
                        y_offset + int(160 * self.scale) + 3, 
                        outline="blue", width=3, tags=item_tag
                    )
                    
                img_id = self.create_image(x_offset, y_offset, anchor=tk.NW, image=img, tags=item_tag)
                self.addtag_withtag(item_tag, img_id)
                
                x_offset += int(170 * self.scale)
                max_row_height = max(max_row_height, int(160 * self.scale))
                
            elif item["type"] == "video":
                # Draw video thumbnail with icon
                if is_selected:
                    rect_id = self.create_rectangle(
                        x_offset - 3, y_offset - 3, 
                        x_offset + int(160 * self.scale) + 3, 
                        y_offset + int(160 * self.scale) + 3, 
                        outline="blue", width=3, tags=item_tag
                    )
                
                # Draw video icon
                self.create_rectangle(
                    x_offset, y_offset, 
                    x_offset + int(160 * self.scale), 
                    y_offset + int(160 * self.scale),
                    fill="#e0e0e0", outline="#c0c0c0", tags=item_tag
                )
                
                # Show video preview if available
                if "preview" in item and item["preview"]:
                    img_id = self.create_image(
                        x_offset, y_offset, 
                        anchor=tk.NW, image=item["preview"], 
                        tags=item_tag
                    )
                
                # Add play button overlay
                play_btn_x = x_offset + int(80 * self.scale)
                play_btn_y = y_offset + int(80 * self.scale)
                self.create_oval(
                    play_btn_x - 20, play_btn_y - 20,
                    play_btn_x + 20, play_btn_y + 20,
                    fill="#80c080", outline="#60a060", 
                    tags=item_tag
                )
                
                # Triangle play symbol
                self.create_polygon(
                    play_btn_x - 5, play_btn_y - 10,
                    play_btn_x - 5, play_btn_y + 10,
                    play_btn_x + 10, play_btn_y,
                    fill="white", tags=item_tag
                )
                
                self.create_text(x_offset + int(80 * self.scale), 
                               y_offset + int(140 * self.scale),
                               text="视频", anchor=tk.CENTER,
                               font=("TkDefaultFont", int(12 * self.scale)),
                               tags=item_tag)
                               
                x_offset += int(170 * self.scale)
                max_row_height = max(max_row_height, int(160 * self.scale))
            else:
                self.create_text(
                    x_offset + int(80 * self.scale), 
                    y_offset + int(80 * self.scale),
                    text=item["preview"], anchor=tk.CENTER,
                    font=("TkDefaultFont", int(12 * self.scale)),
                    tags=item_tag
                )
                x_offset += int(170 * self.scale)
                max_row_height = max(max_row_height, int(80 * self.scale))
            
            if x_offset > self.winfo_width() - 200:
                x_offset = 10
                y_offset += max_row_height + 20
                max_row_height = 0

    def get_selected_item(self):
        """Return the currently selected media item"""
        if self.selected_item is not None and 0 <= self.selected_item < len(self.items):
            return self.items[self.selected_item]
        return None

class HistoryManager:
    """Manages conversation and analysis history"""
    def __init__(self, save_dir="history"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.current_session = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_entry(self, entry):
        self.current_session.append({
            "timestamp": datetime.now().isoformat(),
            "content": entry
        })

    def save_session(self):
        import pickle
        file_path = self.save_dir / f"session_{self.session_id}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(self.current_session, f)
        
    def load_session(self, session_id):
        import pickle
        file_path = self.save_dir / f"session_{session_id}.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                self.current_session = pickle.load(f)
        return self.current_session

    def export_csv(self, file_path):
        import csv
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Content"])
            for entry in self.current_session:
                writer.writerow([entry["timestamp"], str(entry["content"])])

class VisualizationPanel(tk.Frame):
    """Visualization panel for annotated images and videos"""
    def __init__(self, parent):
        super().__init__(parent, bg="#f8f9fa")
        self.pack_propagate(False)
        
        # Create controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, side=tk.TOP, pady=5)
        
        ttk.Button(control_frame, text="保存可视化", command=self.save_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="导出全部", command=self.export_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="上一个", command=self.show_previous).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="下一个", command=self.show_next).pack(side=tk.LEFT, padx=2)
        
        # Label for stats
        self.stat_label = ttk.Label(control_frame, text="0/0 项")
        self.stat_label.pack(side=tk.RIGHT, padx=5)
        
        # Create canvas for visualization
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar_y = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Visualization data
        self.visualizations = []
        self.current_index = -1
        self.current_image = None
        
        # Video frame display
        self.video_frame = ttk.Frame(self.canvas_frame)
        self.video_label = ttk.Label(self.video_frame)
        self.video_frame.pack_forget()  # Hidden initially
    
    def add_visualization(self, image, metadata=None):
        """Add a new visualization to the panel"""
        # Generate unique ID for this visualization
        vis_id = str(uuid.uuid4())
        
        # If image is a numpy array (OpenCV), convert to PIL Image
        if isinstance(image, np.ndarray):
            # Convert from BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return None
        
        self.visualizations.append({
            "id": vis_id,
            "image": pil_image, 
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        
        # Show the newly added visualization
        self.current_index = len(self.visualizations) - 1
        self.show_current()
        
        return vis_id
    
    def add_video_frame(self, frame, timestamp=None, metadata=None):
        """Add a video frame for display (no storage)"""
        if not isinstance(frame, np.ndarray):
            logger.error("Video frame must be a numpy array")
            return
            
        # Convert from BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(frame_rgb)
        photo_image = ImageTk.PhotoImage(pil_image)
        
        # Update video label
        self.video_label.configure(image=photo_image)
        self.video_label.image = photo_image  # Keep a reference
        
        # Hide canvas and show video frame
        self.canvas.pack_forget()
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Update stats
        frame_info = f"视频帧 | "
        if timestamp is not None:
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            frame_info += f"时间: {mins:02d}:{secs:02d} | "
            
        if metadata:
            if "tracks" in metadata:
                frame_info += f"跟踪目标: {len(metadata['tracks'])} | "
            if "detections" in metadata and metadata["detections"]:
                frame_info += f"检测目标: {len(metadata['detections'])}"
                
        self.stat_label.config(text=frame_info)
    
    def show_current(self):
        """Show current visualization"""
        # Switch back to image mode
        self.video_frame.pack_forget()
        self.canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        if not self.visualizations or self.current_index < 0:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="无可视化数据", font=("TkDefaultFont", 14)
            )
            self.stat_label.config(text="0/0 项")
            return
        
        vis = self.visualizations[self.current_index]
        
        # Update stats display
        self.stat_label.config(text=f"{self.current_index + 1}/{len(self.visualizations)} 项")
        
        # Clear canvas and show image
        self.canvas.delete("all")
        
        # Convert PIL image to PhotoImage
        self.current_image = ImageTk.PhotoImage(vis["image"])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        
        # Set canvas scrollregion
        self.canvas.config(
            scrollregion=(0, 0, vis["image"].width, vis["image"].height)
        )
        
        # Show metadata if available
        if vis["metadata"]:
            info_text = "\n".join(f"{k}: {v}" for k, v in vis["metadata"].items())
            self.canvas.create_text(
                10, 10, text=info_text, anchor=tk.NW,
                font=("TkDefaultFont", 12), fill="white", 
                tags="metadata"
            )
            
            # Add shadow effect for better readability
            bbox = self.canvas.bbox("metadata")
            if bbox:
                padding = 5
                self.canvas.create_rectangle(
                    bbox[0]-padding, bbox[1]-padding,
                    bbox[2]+padding, bbox[3]+padding,
                    fill="black", stipple="gray25", outline="",
                    tags="metadata_bg"
                )
                self.canvas.tag_lower("metadata_bg", "metadata")
    
    def show_next(self):
        """Show next visualization"""
        if not self.visualizations:
            return
        
        self.current_index = (self.current_index + 1) % len(self.visualizations)
        self.show_current()
    
    def show_previous(self):
        """Show previous visualization"""
        if not self.visualizations:
            return
        
        self.current_index = (self.current_index - 1) % len(self.visualizations)
        self.show_current()
    
    def save_current(self):
        """Save current visualization to file"""
        # Check if we're in video mode
        if self.video_frame.winfo_ismapped():
            # Capture current video frame
            if hasattr(self.video_label, 'image') and self.video_label.image:
                # Get PIL image from PhotoImage
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                    initialfile=f"video_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                
                if file_path:
                    try:
                        # Get PIL image and save it
                        img = ImageTk._PhotoImage__photo(self.video_label.image)
                        img.write(file_path)
                        messagebox.showinfo("成功", f"视频帧已保存至: {file_path}")
                        return
                    except Exception as e:
                        messagebox.showerror("错误", f"保存失败: {str(e)}")
                        return
            
            messagebox.showinfo("提示", "无可保存的视频帧")
            return
            
        # Otherwise process normal visualizations
        if not self.visualizations or self.current_index < 0:
            messagebox.showwarning("警告", "无可视化数据可保存")
            return
        
        vis = self.visualizations[self.current_index]
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile=f"vis_{vis['id'][:8]}.png"
        )
        
        if file_path:
            try:
                vis["image"].save(file_path)
                messagebox.showinfo("成功", f"可视化已保存至: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def export_all(self):
        """Export all visualizations to a directory"""
        if not self.visualizations:
            messagebox.showwarning("警告", "无可视化数据可导出")
            return
        
        export_dir = filedialog.askdirectory(title="选择导出文件夹")
        if not export_dir:
            return
            
        export_dir = Path(export_dir)
        export_dir.mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = []
        
        # Export all images
        for i, vis in enumerate(self.visualizations):
            try:
                file_name = f"vis_{vis['id'][:8]}_{i:03d}.png"
                file_path = export_dir / file_name
                
                # Save image
                vis["image"].save(file_path)
                
                # Add to metadata
                metadata.append({
                    "id": vis["id"],
                    "file": file_name,
                    "timestamp": vis["timestamp"],
                    "metadata": vis["metadata"]
                })
                
            except Exception as e:
                logger.error(f"Error exporting visualization {i}: {str(e)}")
                continue
        
        # Save metadata file
        try:
            with open(export_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            messagebox.showinfo("成功", f"共导出 {len(metadata)} 项可视化")
        except Exception as e:
            messagebox.showerror("错误", f"导出元数据失败: {str(e)}")
    
    def clear_all(self):
        """Clear all visualizations"""
        if not self.visualizations:
            return
            
        if messagebox.askyesno("确认", "是否清除所有可视化?"):
            self.visualizations = []
            self.current_index = -1
            self.show_current()

class VideoProcessor:
    """Process videos for AI analysis and tracking"""
    
    def __init__(self, client=None, model=None, config=None):
        self.client = client
        self.model_name = model or list(SUPPORTED_MODELS.keys())[0]
        self.config = config or {}
        
        # Create tracker
        self.tracker_algorithm = self.config.get("tracking_algorithm", "CSRT")
        self.create_tracker()
        
        # Processing state
        self.is_processing = False
        self.current_video = None
        self.frame_count = 0
        self.current_frame = None
        self.fps = 0
        
        # Results
        self.detections = []
        self.tracks = {}
        self.track_history = {}
        
        # Thread management
        self.processing_thread = None
        self.detection_interval = self.config.get("detection_interval", 20)  # Run detection every 20 frames
        self.result_queue = Queue()
        
    def create_tracker(self):
        """Create appropriate tracker using tracking_utils"""
        from tracking_utils import create_tracker, get_available_algorithms, track_and_draw, create_yolo_tracker
        
        # 获取可用算法
        available_algorithms = get_available_algorithms()
        logger.info(f"可用的跟踪算法: {', '.join(available_algorithms)}")
        
        # 检查YOLO是否可用
        yolo_available = os.environ.get("YOLO_AVAILABLE", "0") == "1"
        if yolo_available and "YOLO" not in available_algorithms:
            # YOLO环境变量存在但跟踪算法中没有YOLO，尝试手动添加
            logger.info("检测到YOLO环境变量，尝试添加YOLO跟踪器")
            available_algorithms.append("YOLO")
        
        # 如果用户明确要求使用YOLO但不可用，记录警告
        if self.tracker_algorithm == "YOLO" and not yolo_available:
            logger.warning("请求YOLO跟踪器但YOLO未启用，将使用智能跟踪器代替")
            self.tracker_algorithm = "智能" if "智能" in available_algorithms else "Basic"
        
        # 检查请求的算法是否可用
        if self.tracker_algorithm not in available_algorithms:
            logger.warning(f"请求的跟踪器 {self.tracker_algorithm} 不可用，使用 Basic")
            self.tracker_algorithm = "Basic"
        
        # 创建一个返回跟踪器的函数
        if self.tracker_algorithm == "YOLO" and yolo_available:
            # 对YOLO跟踪器使用特殊的创建函数
            self.tracker_create_func = lambda: create_yolo_tracker()
            logger.info(f"已设置YOLO跟踪器")
        else:
            # 创建常规跟踪器，同时传递默认标签
            self.tracker_create_func = lambda: create_tracker(
                algorithm=self.tracker_algorithm, 
                label=getattr(self, "default_label", "目标")
            )
            logger.info(f"已设置跟踪器: {self.tracker_algorithm}")
        
        # 添加跟踪器信息到界面
        try:
            if hasattr(self, "tracker_info_label"):
                self.tracker_info_label.setText(f"当前跟踪器: {self.tracker_algorithm}")
        except Exception as e:
            logger.error(f"更新跟踪器信息标签出错: {e}")
        

            
    def load_video(self, video_path):
        """Load video file for processing"""
        try:
            self.current_video = cv2.VideoCapture(video_path)
            if not self.current_video.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False
                
            # Get video properties
            self.frame_count = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.current_video.get(cv2.CAP_PROP_FPS)
            width = int(self.current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Loaded video: {Path(video_path).name}, {width}x{height}, {self.frame_count} frames, {self.fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            return False
    
    def reset(self):
        """Reset processor state"""
        if self.is_processing:
            self.stop_processing()
            
        if self.current_video and self.current_video.isOpened():
            self.current_video.release()
            
        self.current_video = None
        self.current_frame = None
        self.frame_count = 0
        self.fps = 0
        self.detections = []
        self.tracks = {}
        self.track_history = {}
        
    def get_frame(self, frame_number=None):
        """Get a specific frame from the video"""
        if not self.current_video or not self.current_video.isOpened():
            return None
            
        if frame_number is not None:
            self.current_video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
        ret, frame = self.current_video.read()
        if ret:
            self.current_frame = frame
            return frame
        return None
    
    def start_processing(self, callback=None):
        """Start video processing in a separate thread"""
        if self.is_processing:
            logger.warning("Video processing already running")
            return False
            
        if not self.current_video or not self.current_video.isOpened():
            logger.error("No video loaded")
            return False
            
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._process_video,
            args=(callback,)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        return True
    
    def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
    
    def _process_video(self, callback=None):
        """Process video frames with AI and tracking"""
        if not self.current_video or not self.current_video.isOpened():
            logger.error("No video loaded for processing")
            self.is_processing = False
            return
            
        # Reset video to beginning
        self.current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_index = 0
        last_detection_frame = -self.detection_interval  # Ensure first frame gets detection
        
        # Initialize trackers and track history
        self.tracks = {}
        self.track_history = {}
        trackers = []
        next_track_id = 0
        
        while self.is_processing:
            start_time = time.time()
            
            ret, frame = self.current_video.read()
            if not ret:
                logger.info("End of video reached")
                break
                
            self.current_frame = frame
            
            # Should we run detection on this frame?
            run_detection = (frame_index - last_detection_frame) >= self.detection_interval
            
            # Update existing trackers
            tracks_to_remove = []
            for track_id, tracker in self.tracks.items():
                success = False
                try:
                    success, bbox = tracker.update(frame)
                except Exception as e:
                    logger.error(f"Error updating tracker {track_id}: {e}")
                    success = False
                    
                if success:
                    # Update track history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    # Convert bbox to integer coordinates
                    bbox = tuple(int(v) for v in bbox)
                    self.track_history[track_id].append(bbox)
                    
                    # Keep track history to a reasonable size
                    max_history = 30  # Max number of points to keep
                    if len(self.track_history[track_id]) > max_history:
                        self.track_history[track_id] = self.track_history[track_id][-max_history:]
                else:
                    # Mark tracker for removal
                    tracks_to_remove.append(track_id)
                    
            # Remove failed trackers
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                
            # Run detection with AI if needed
            detection_annotations = []
            if run_detection:
                # Run detection with the AI model
                logger.debug(f"Running detection on frame {frame_index}")
                detection_annotations = self._detect_objects(frame)
                
                if detection_annotations:
                    last_detection_frame = frame_index
                    self.detections = detection_annotations
                    
                    # Create new trackers for each detection
                    for detection in detection_annotations:
                        bbox = detection.get("bbox_2d")
                        if not bbox or len(bbox) != 4:
                            continue
                            
                        # Convert from [x1, y1, x2, y2] to [x, y, w, h] format for tracker
                        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                        
                        # Create a new tracker if supported
                        if self.tracker_create_func:
                            try:
                                tracker = self.tracker_create_func()
                                success = tracker.init(frame, (x, y, w, h))
                                
                                if success:
                                    track_id = next_track_id
                                    next_track_id += 1
                                    self.tracks[track_id] = tracker
                                    self.track_history[track_id] = [(x, y, w, h)]
                            except Exception as e:
                                logger.error(f"Error creating tracker: {e}")
            
            # Create visualizations
            original_vis = frame.copy()
            tracking_vis = self._visualize_tracking(frame)
                                             
            # If we have detections, also visualize those
            if run_detection and self.detections:
                detection_vis = self._visualize_detection(frame, self.detections)
            else:
                detection_vis = None
            
            # Create result mosaic if we have detection results
            if detection_vis is not None:
                # Create 2x2 grid with original, detection, tracking frames
                frames = [original_vis, detection_vis, tracking_vis]
                titles = ["Original", "AI Detection", "Object Tracking"]
                mosaic = self._create_video_mosaic(frames, titles, (2, 2))
            else:
                # Just show original and tracking if no detection was run
                frames = [original_vis, tracking_vis]
                titles = ["Original", "Object Tracking"]  
                mosaic = self._create_video_mosaic(frames, titles, (2, 1))
            
            # Put results in queue for the UI to consume
            result = {
                "frame_index": frame_index,
                "timestamp": frame_index / self.fps if self.fps > 0 else 0,
                "tracks": self.tracks,
                "detections": self.detections if run_detection else None,
                "visualizations": {
                    "mosaic": mosaic,
                    "tracking": tracking_vis,
                    "detection": detection_vis if run_detection else None
                },
                "annotations": [
                    # Create annotations from tracked bounding boxes
                    {"bbox_2d": [x, y, x+w, y+h], "track_id": track_id, "label": f"Object {track_id}"}
                    for track_id, bbox in self.tracks.items()
                ]
            }
            
            self.result_queue.put(result)
            
            # Call callback if provided
            if callback:
                callback(result)
                
            # Control processing speed
            frame_index += 1
            elapsed = time.time() - start_time
            delay = max(0, 1.0/30.0 - elapsed)  # Cap at 30 FPS for UI
            time.sleep(delay)
            
        # Mark as done
        self.is_processing = False
    
    def _detect_objects(self, frame):
        """Use AI model to detect objects in a frame"""
        if not self.client:
            logger.error("No API client available for detection")
            return []
            
        try:
            # Convert frame to base64 for API
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(img_encoded).decode()
            img_data_uri = "data:image/jpeg;base64," + img_base64
            
            # Prepare messages for API
            prompt = self.config.get("detection_prompt", "请分析图像并识别其中的目标，返回边界框和类别标签")
            messages = [
                {
                    "role": "system",
                    "content": "请分析图像并识别目标对象。返回JSON格式，包含annotations数组，每个标注包括bbox_2d: [x1, y1, x2, y2]和label字段。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_data_uri}}
                    ]
                }
            ]
            
            # Make API call
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 2048)
            )
            
            # Parse the response
            response_text = completion.choices[0].message.content
            annotations = self._extract_annotations_from_response(response_text)
            
            logger.debug(f"Detected {len(annotations)} objects")
            return annotations
            
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return []
    
    def _extract_annotations_from_response(self, response_text):
        """Extract bbox annotations from API response text"""
        try:
            # Look for JSON content in the response
            import re
            json_pattern = re.compile(r'\{.*\}', re.DOTALL)
            json_match = json_pattern.search(response_text)
            
            if json_match:
                json_str = json_match.group()
                json_data = json.loads(json_str)
                
                if "annotations" in json_data:
                    return json_data["annotations"]
                    
            logger.warning("No valid annotations found in response")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting annotations: {str(e)}")
            return []
            
    def get_next_result(self, timeout=0.1):
        """Get next processing result from queue"""
        try:
            return self.result_queue.get(timeout=timeout)
        except Exception:
            return None
    
    def _visualize_tracking(self, frame):
        """Visualize object tracks with trajectory lines"""
        vis_frame = frame.copy()
        
        # Get visualization settings from config
        color_scheme = self.config.get("visualization", {}).get("color_scheme", "default")
        
        # Define color schemes
        color_schemes = {
            "default": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)],
            "rainbow": [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128)],
            "pastel": [(255, 179, 186), (255, 223, 186), (255, 255, 186), (186, 255, 201), (186, 225, 255), (218, 186, 255)],
            "highvis": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        }
        
        colors = color_schemes.get(color_scheme, color_schemes["default"])
        
        # Draw tracks and histories
        for track_id, tracker in self.tracks.items():
            try:
                # Get the latest position
                success, bbox = tracker.update(frame.copy())  # Use a copy to avoid modifying the frame
                
                if success:
                    # Get color for this track
                    color_idx = track_id % len(colors)
                    color = colors[color_idx]
                    
                    # Convert to integers
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw track ID
                    cv2.putText(vis_frame, f"ID: {track_id}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw trajectory if we have history
                    history = self.track_history.get(track_id, [])
                    if len(history) > 1:
                        # Draw lines connecting history points
                        for i in range(1, len(history)):
                            pt1_x, pt1_y = history[i-1][0] + history[i-1][2]//2, history[i-1][1] + history[i-1][3]//2
                            pt2_x, pt2_y = history[i][0] + history[i][2]//2, history[i][1] + history[i][3]//2
                            cv2.line(vis_frame, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
            except Exception as e:
                logger.error(f"Error visualizing track {track_id}: {e}")
                
        return vis_frame
        
    def _visualize_detection(self, frame, annotations):
        """Visualize detection results"""
        vis_frame = frame.copy()
        
        # Get visualization settings from config
        show_labels = self.config.get("visualization", {}).get("show_labels", True)
        show_scores = self.config.get("visualization", {}).get("show_scores", False)
        color_scheme = self.config.get("visualization", {}).get("color_scheme", "default")
        
        # Define color schemes
        color_schemes = {
            "default": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)],
            "rainbow": [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128)],
            "pastel": [(255, 179, 186), (255, 223, 186), (255, 255, 186), (186, 255, 201), (186, 225, 255), (218, 186, 255)],
            "highvis": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        }
        
        colors = color_schemes.get(color_scheme, color_schemes["default"])
        
        # Map class names to colors for consistency
        class_colors = {}
        
        for i, anno in enumerate(annotations):
            bbox = anno.get("bbox_2d")
            label = anno.get("label", "Unknown")
            confidence = anno.get("confidence")
            
            if not bbox or len(bbox) != 4:
                continue
                
            # Get color for this class
            if label not in class_colors:
                class_colors[label] = colors[len(class_colors) % len(colors)]
            color = class_colors[label]
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and confidence
            if show_labels:
                text = label
                if show_scores and confidence:
                    conf_val = float(confidence) if isinstance(confidence, (int, float, str)) else 0.0
                    text = f"{label}: {conf_val:.2f}"
                    
                # Add background for text for better visibility
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                cv2.putText(vis_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        return vis_frame
        
    def _create_video_mosaic(self, frames, titles=None, grid_size=None):
        """Create a mosaic of multiple video frames"""
        if not frames:
            return None
            
        # Determine grid size if not provided
        if grid_size is None:
            n_frames = len(frames)
            grid_width = int(np.ceil(np.sqrt(n_frames)))
            grid_height = int(np.ceil(n_frames / grid_width))
            grid_size = (grid_width, grid_height)
        else:
            grid_width, grid_height = grid_size
        
        # Ensure all frames have the same size by resizing if necessary
        frame_h, frame_w = frames[0].shape[:2]
        frame_size = (frame_w, frame_h)
        
        for i in range(1, len(frames)):
            if frames[i].shape[:2] != (frame_h, frame_w):
                frames[i] = cv2.resize(frames[i], frame_size)
        
        # Create empty mosaic canvas
        mosaic = np.zeros((frame_h * grid_height, frame_w * grid_width, 3), dtype=np.uint8)
        
        # Fill in the mosaic
        frame_idx = 0
        for i in range(grid_height):
            for j in range(grid_width):
                if frame_idx < len(frames):
                    y_start = i * frame_h
                    y_end = y_start + frame_h
                    x_start = j * frame_w
                    x_end = x_start + frame_w
                    
                    mosaic[y_start:y_end, x_start:x_end] = frames[frame_idx]
                    
                    # Add title if provided
                    if titles and frame_idx < len(titles):
                        title = titles[frame_idx]
                        cv2.putText(mosaic, title, 
                                   (x_start + 10, y_start + 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    frame_idx += 1
                else:
                    break
        
        return mosaic
    
    def export_tracking_results(self, output_path):
        """Export tracking results to file"""
        try:
            results = {
                "tracks": {},
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "tracker_algorithm": self.tracker_algorithm
                }
            }
            
            # Convert track history to list for JSON serialization
            for track_id, history in self.track_history.items():
                results["tracks"][str(track_id)] = {
                    "history": [list(bbox) for bbox in history]
                }
                
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Tracking results exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting tracking results: {str(e)}")
            return False

class MultimodalApp:
    """Main application class for multimodal analysis with video tracking"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("智能多模态分析系统 v4.0")
        self.root.geometry("1400x900")
        self.root.option_add("*Font", "TkDefaultFont 12")
        
        # Initialize UI variables first
        self.model_var = tk.StringVar(value=list(SUPPORTED_MODELS.keys())[0])
        self.temperature_var = tk.DoubleVar(value=DEFAULT_CONFIG["temperature"])
        
        # YOLO相关初始化
        self.yolo_available = os.environ.get("YOLO_AVAILABLE", "0") == "1"
        self.yolo_model_var = tk.StringVar(value="yolov12s")  # 默认使用小型模型平衡性能和精度
        self.yolo_detector = None
        self.yolo_tracking_active = False
        self.using_yolo_tracker = False
        
        # 如果YOLO可用，初始化YOLO检测器
        if self.yolo_available:
            try:
                logger.info("正在初始化YOLO检测器...")
                from tracking_utils import init_yolo_detector
                self.yolo_detector = init_yolo_detector(self.yolo_model_var.get())
                if self.yolo_detector.initialized:
                    logger.info("YOLO检测器初始化成功")
                else:
                    logger.warning("YOLO检测器初始化失败")
                    self.yolo_available = False
            except Exception as e:
                logger.error(f"初始化YOLO检测器时出错: {e}")
                self.yolo_available = False
        
        # Get available trackers (after YOLO initialization to include it in options)
        available_trackers = tracking_utils.get_available_algorithms()
        
        # 如果YOLO可用且已初始化成功，优先使用YOLO跟踪器
        default_tracker = "YOLO" if (self.yolo_available and "YOLO" in available_trackers) else \
                        (available_trackers[0] if available_trackers else "Basic")
                        
        self.tracking_algo_var = tk.StringVar(value=default_tracker)
        if self.tracking_algo_var.get() == "YOLO":
            self.using_yolo_tracker = True
        
        self.show_labels_var = tk.BooleanVar(value=DEFAULT_CONFIG["visualization"]["show_labels"])
        self.show_scores_var = tk.BooleanVar(value=DEFAULT_CONFIG["visualization"]["show_scores"])
        self.color_scheme_var = tk.StringVar(value=DEFAULT_CONFIG["visualization"]["color_scheme"])
        self.batch_mode = tk.BooleanVar(value=False)
        
        # Initialize other components
        self.output_text = None  # Will be set in create_output_panel
        self.visualization_panel = None  # Will be set in create_output_panel
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            api_key = simpledialog.askstring(
                "API Key", "请输入 DashScope API Key:",
                parent=self.root
            )
            if not api_key:
                messagebox.showerror("错误", "未提供 API Key，应用将退出")
                self.root.quit()
                return
                
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.history_manager = HistoryManager()
        self.conversation = []
        self.media_items = []
        self.prompt_template = None
        
        # 更新配置以包含YOLO相关设置
        video_processor_config = {
            "temperature": self.temperature_var.get(),
            "tracking_algorithm": self.tracking_algo_var.get(),
            "detection_interval": 30,
            "visualization": {
                "show_labels": self.show_labels_var.get(),
                "show_scores": self.show_scores_var.get(),
                "color_scheme": self.color_scheme_var.get()
            }
        }
        
        # 添加YOLO特定配置
        if self.yolo_available:
            video_processor_config.update({
                "yolo_available": True,
                "yolo_model": self.yolo_model_var.get(),
                "yolo_detector": self.yolo_detector  # 传递已初始化的检测器
            })
        
        # In the __init__ method, when creating the video processor:
        self.video_processor = VideoProcessor(
            client=self.client,
            model=self.model_var.get(),
            config=video_processor_config
        )
        
        # UI state for video playback
        self.is_playing_video = False
        self.video_play_thread = None
        
        # Build interface
        self.create_menu()
        self.create_ui()
        self.style_widgets()
        
        # 如果YOLO可用，创建YOLO控制面板
        if self.yolo_available:
            self.create_yolo_controls()
        
        # Start UI update thread for video processing
        self.start_ui_update_thread()

    def create_yolo_controls(self):
        """创建YOLO控制面板，使用单独的窗口，具有完整的视频动态追踪功能"""
        
        # ---------- 内部函数定义 ----------
        def init_video_source(source_path=None, camera_index=0):
            """初始化视频源（文件或摄像头）"""
            nonlocal cap, video_source_path, is_camera
            
            # 关闭现有视频源
            if cap is not None:
                cap.release()
            
            try:
                if source_path:
                    # 使用文件作为视频源
                    cap = cv2.VideoCapture(source_path)
                    video_source_path = source_path
                    is_camera = False
                    source_label.config(text=f"视频源: {os.path.basename(source_path)}")
                else:
                    # 使用摄像头作为视频源
                    cap = cv2.VideoCapture(camera_index)
                    video_source_path = None
                    is_camera = True
                    source_label.config(text=f"视频源: 摄像头 #{camera_index}")
                
                if not cap.isOpened():
                    raise Exception("无法打开视频源")
                    
                # 获取视频信息
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or math.isnan(fps):
                    fps = 30  # 默认帧率
                    
                frame_delay = int(1000 / fps)  # 毫秒
                
                # 更新状态
                status_label.config(text="就绪")
                return True
                
            except Exception as e:
                error_msg = f"初始化视频源失败: {str(e)}"
                status_label.config(text=error_msg)
                logger.error(error_msg)
                return False
        
        def select_video_file():
            """选择视频文件"""
            file_path = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[
                    ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                    ("所有文件", "*.*")
                ]
            )
            
            if file_path:
                # 停止当前追踪
                stop_tracking()
                # 初始化新的视频源
                if init_video_source(file_path):
                    # 启用开始追踪按钮
                    start_tracking_btn.config(state=tk.NORMAL)
        
        def toggle_camera():
            """切换摄像头视频源"""
            nonlocal is_camera
            
            if is_camera:
                # 如果当前是摄像头，则停止
                stop_tracking()
                camera_btn.config(text="打开摄像头")
                is_camera = False
                status_label.config(text="已关闭摄像头")
            else:
                # 停止当前追踪
                stop_tracking()
                # 初始化摄像头
                if init_video_source(camera_index=0):
                    camera_btn.config(text="关闭摄像头")
                    # 启用开始追踪按钮
                    start_tracking_btn.config(state=tk.NORMAL)
        
        def update_model():
            """更新YOLO模型"""
            nonlocal yolo_detector
            
            model_name = model_var.get()
            status_label.config(text=f"正在加载 {model_name} 模型...")
            
            try:
                from tracking_utils import init_yolo_detector
                yolo_detector = init_yolo_detector(model_name)
                
                if yolo_detector and yolo_detector.initialized:
                    logger.info(f"成功加载模型: {model_name}")
                    status_label.config(text=f"已加载 {model_name}")
                    model_status_label.config(text="模型状态: 已加载", foreground="green")
                    return True
                else:
                    error_msg = f"模型 {model_name} 初始化失败"
                    status_label.config(text=error_msg)
                    model_status_label.config(text="模型状态: 加载失败", foreground="red")
                    logger.error(error_msg)
                    return False
                    
            except Exception as e:
                error_msg = f"加载模型出错: {str(e)}"
                status_label.config(text=error_msg)
                model_status_label.config(text="模型状态: 加载失败", foreground="red")
                logger.error(error_msg)
                return False
        
        def start_tracking():
            """开始追踪视频目标"""
            nonlocal is_tracking, tracking_paused, roi_selector_active
            
            if is_tracking:
                # 如果已在追踪，暂停/继续
                tracking_paused = not tracking_paused
                if tracking_paused:
                    start_tracking_btn.config(text="继续追踪")
                    status_label.config(text="追踪已暂停")
                else:
                    start_tracking_btn.config(text="暂停追踪")
                    status_label.config(text="追踪继续中")
                return
            
            # 确保视频源已经初始化
            if cap is None or not cap.isOpened():
                status_label.config(text="错误: 请先选择视频或打开摄像头")
                return
                
            # 确保模型已经加载
            if yolo_detector is None or not yolo_detector.initialized:
                if not update_model():
                    return
            
            # 读取第一帧用于ROI选择
            ret, frame = cap.read()
            if not ret:
                status_label.config(text="无法读取视频帧")
                return
                
            # 重置视频到开始位置 (如果不是摄像头)
            if not is_camera:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            # 允许用户选择ROI
            roi_selector_active = True
            status_label.config(text="请在视频上选择追踪目标")
            
            # 调整帧大小用于显示
            frame = resize_frame(frame)
            
            # 显示帧并等待用户选择ROI
            cv2.namedWindow("选择追踪目标", cv2.WINDOW_NORMAL)
            roi = cv2.selectROI("选择追踪目标", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("选择追踪目标")
            
            roi_selector_active = False
            
            # 检查ROI是否有效
            if roi[2] <= 0 or roi[3] <= 0:
                status_label.config(text="未选择有效目标区域，已取消追踪")
                return
                
            # 创建YOLO追踪器
            try:
                from tracking_utils import create_yolo_tracker
                tracker = create_yolo_tracker(yolo_model=model_var.get())
                
                # 读取新的一帧用于初始化追踪器
                ret, frame = cap.read()
                if not ret:
                    status_label.config(text="无法读取视频帧")
                    return
                    
                # 如果不是摄像头，再次重置视频到开始位置
                if not is_camera:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        status_label.config(text="无法读取视频帧")
                        return
                        
                # 初始化跟踪器
                success = tracker.init(frame, roi)
                if not success:
                    status_label.config(text="初始化追踪器失败")
                    return
                    
                # 设置追踪状态
                is_tracking = True
                tracking_paused = False
                
                # 更新UI
                start_tracking_btn.config(text="暂停追踪")
                stop_tracking_btn.config(state=tk.NORMAL)
                status_label.config(text="正在追踪目标")
                
                # 开始追踪线程
                tracking_thread = threading.Thread(target=tracking_loop, args=(tracker,), daemon=True)
                tracking_thread.start()
                
            except Exception as e:
                error_msg = f"启动追踪时出错: {str(e)}"
                status_label.config(text=error_msg)
                logger.error(error_msg)
        
        def tracking_loop(tracker):
            """追踪主循环（在单独线程中运行）"""
            nonlocal is_tracking, tracking_paused
            
            # 获取视频信息
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or math.isnan(fps):
                    fps = 30  # 默认帧率
                
                frame_delay = 1.0 / fps  # 秒
            else:
                frame_delay = 1.0 / 30  # 默认帧率
            
            # 如果启用了录制，初始化视频写入器
            video_writer = None
            if recording_enabled.get():
                try:
                    # 读取一帧获取尺寸
                    ret, temp_frame = cap.read()
                    if ret:
                        # 重置视频位置
                        if not is_camera:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        
                        height, width = temp_frame.shape[:2]
                        now = datetime.now()
                        timestamp = now.strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(
                            os.path.expanduser("~"),
                            "Videos",
                            f"yolo_tracking_{timestamp}.mp4"
                        )
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # 创建视频写入器
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                        
                        yolo_window.after(0, lambda: status_label.config(text=f"录制到: {output_path}"))
                except Exception as e:
                    error_msg = f"初始化视频录制失败: {str(e)}"
                    logger.error(error_msg)
                    yolo_window.after(0, lambda: status_label.config(text=error_msg))
            
            frame_count = 0
            success_frames = 0
            
            try:
                while is_tracking and cap.isOpened():
                    if tracking_paused:
                        # 暂停时降低CPU使用率
                        time.sleep(0.1)
                        continue
                    
                    # 读取帧
                    ret, frame = cap.read()
                    if not ret:
                        # 如果是文件并到达末尾，循环播放
                        if not is_camera:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                break
                        else:
                            break
                    
                    frame_count += 1
                    
                    # 更新跟踪器
                    success, bbox = tracker.update(frame)
                    
                    # 绘制结果
                    if success:
                        success_frames += 1
                        # 使用跟踪器内置的绘制功能
                        result_frame = tracker.draw_bbox(frame)
                    else:
                        # 追踪失败
                        result_frame = frame.copy()
                        cv2.putText(
                            result_frame, 
                            "跟踪失败", 
                            (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255), 
                            2
                        )
                    
                    # 计算追踪成功率
                    if frame_count > 0:
                        success_rate = (success_frames / frame_count) * 100
                        # 更新UI (使用after确保在主线程中)
                        yolo_window.after(0, lambda rate=success_rate: 
                            tracking_rate_label.config(text=f"追踪成功率: {rate:.1f}%"))
                    
                    # 写入帧到视频文件（如果启用录制）
                    if video_writer is not None:
                        video_writer.write(result_frame)
                    
                    # 调整大小用于显示
                    display_frame = resize_frame(result_frame)
                    
                    # 将OpenCV帧转换为PIL格式
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    img_tk = ImageTk.PhotoImage(image=pil_img)
                    
                    # 更新UI (使用after确保在主线程中)
                    yolo_window.after(0, lambda img=img_tk: update_preview(img))
                    
                    # 控制帧率
                    time.sleep(frame_delay)
            
            except Exception as e:
                error_msg = f"追踪过程中出错: {str(e)}"
                logger.error(error_msg)
                import traceback
                traceback.print_exc()
                yolo_window.after(0, lambda: status_label.config(text=error_msg))
            
            finally:
                # 清理资源
                if video_writer is not None:
                    video_writer.release()
                
                # 重置UI
                yolo_window.after(0, reset_tracking_ui)
        
        def stop_tracking():
            """停止追踪"""
            nonlocal is_tracking, tracking_paused, tracker
            
            is_tracking = False
            tracking_paused = False
            tracker = None
            
            # 重置UI
            reset_tracking_ui()
        
        def reset_tracking_ui():
            """重置追踪UI状态"""
            start_tracking_btn.config(text="开始追踪", state=tk.NORMAL)
            stop_tracking_btn.config(state=tk.DISABLED)
            status_label.config(text="就绪")
            tracking_rate_label.config(text="追踪成功率: --")
        
        def resize_frame(frame, max_width=640, max_height=480):
            """调整帧大小以适应显示区域"""
            height, width = frame.shape[:2]
            
            # 计算缩放比例
            scale = min(max_width / width, max_height / height)
            
            # 如果小于最大尺寸，无需缩放
            if width <= max_width and height <= max_height:
                return frame
            
            # 计算新尺寸
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 调整大小
            resized = cv2.resize(frame, (new_width, new_height))
            return resized
        
        def update_preview(img_tk):
            """更新预览画面"""
            # 清除画布并显示新图像
            preview_canvas.delete("all")
            # 保存引用以防垃圾回收
            preview_canvas.image = img_tk
            preview_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        
        def toggle_recording():
            """切换录制状态"""
            if recording_enabled.get():
                record_btn.config(text="停止录制", background="#ff6666")
            else:
                record_btn.config(text="开始录制", background="#66cc66")
        
        def on_window_close():
            """窗口关闭时的处理"""
            nonlocal is_tracking, cap
            
            # 停止追踪
            is_tracking = False
            
            # 关闭视频源
            if cap is not None:
                cap.release()
            
            # 销毁窗口
            yolo_window.destroy()
        
        # ---------- 窗口创建 ----------
        # 创建主窗口
        yolo_window = tk.Toplevel(self.root)
        yolo_window.title("YOLO 视频追踪控制")
        yolo_window.geometry("800x600")
        yolo_window.protocol("WM_DELETE_WINDOW", on_window_close)
        
        # 状态变量
        cap = None               # 视频捕获对象
        video_source_path = None # 视频文件路径
        is_camera = False        # 是否使用摄像头
        is_tracking = False      # 是否正在追踪
        tracking_paused = False  # 是否暂停追踪
        roi_selector_active = False # 是否正在选择ROI
        tracker = None           # 跟踪器对象
        yolo_detector = None     # YOLO检测器
        
        # Model selection
        model_var = tk.StringVar(value=self.yolo_model_var.get())
        recording_enabled = tk.BooleanVar(value=False)
        
        # 控件背景色
        bg_color = "#f0f0f0"
        
        # ---------- UI布局 ----------
        # 创建主框架
        main_frame = ttk.Frame(yolo_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 上部控制区域
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="5")
        control_frame.pack(fill=tk.X, pady=5)
        
        # 模型选择
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="YOLO模型:").pack(side=tk.LEFT, padx=5)
        models = ["yolov12n", "yolov12s", "yolov12m", "yolov12l", "yolov12x"]
        model_dropdown = ttk.Combobox(model_frame, textvariable=model_var,
                                    values=models, width=10, state="readonly")
        model_dropdown.pack(side=tk.LEFT, padx=5)
        model_dropdown.bind("<<ComboboxSelected>>", lambda e: update_model())
        
        model_status_label = ttk.Label(model_frame, text="模型状态: 未加载")
        model_status_label.pack(side=tk.LEFT, padx=10)
        
        load_model_btn = ttk.Button(model_frame, text="加载模型", command=update_model)
        load_model_btn.pack(side=tk.LEFT, padx=5)
        
        # 视频源选择
        source_frame = ttk.Frame(control_frame)
        source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(source_frame, text="视频源:").pack(side=tk.LEFT, padx=5)
        select_video_btn = ttk.Button(source_frame, text="选择视频文件", command=select_video_file)
        select_video_btn.pack(side=tk.LEFT, padx=5)
        
        camera_btn = ttk.Button(source_frame, text="打开摄像头", command=toggle_camera)
        camera_btn.pack(side=tk.LEFT, padx=5)
        
        source_label = ttk.Label(source_frame, text="未选择视频源")
        source_label.pack(side=tk.LEFT, padx=10)
        
        # 追踪控制
        tracking_frame = ttk.Frame(control_frame)
        tracking_frame.pack(fill=tk.X, pady=5)
        
        start_tracking_btn = ttk.Button(tracking_frame, text="开始追踪", command=start_tracking, state=tk.DISABLED)
        start_tracking_btn.pack(side=tk.LEFT, padx=5)
        
        stop_tracking_btn = ttk.Button(tracking_frame, text="停止追踪", command=stop_tracking, state=tk.DISABLED)
        stop_tracking_btn.pack(side=tk.LEFT, padx=5)
        
        # 录制控制
        record_frame = ttk.Frame(control_frame)
        record_frame.pack(fill=tk.X, pady=5)
        
        record_check = ttk.Checkbutton(record_frame, text="启用录制", variable=recording_enabled, command=toggle_recording)
        record_check.pack(side=tk.LEFT, padx=5)
        
        record_btn = tk.Button(record_frame, text="开始录制", bg="#66cc66", command=toggle_recording)
        record_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态区域
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="状态:").pack(side=tk.LEFT, padx=5)
        status_label = ttk.Label(status_frame, text="就绪")
        status_label.pack(side=tk.LEFT, padx=5)
        
        tracking_rate_label = ttk.Label(status_frame, text="追踪成功率: --")
        tracking_rate_label.pack(side=tk.RIGHT, padx=10)
        
        # 预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="视频预览", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        preview_canvas = tk.Canvas(preview_frame, bg="black", width=640, height=480)
        preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        statusbar_frame = ttk.Frame(main_frame)
        statusbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        info_label = ttk.Label(statusbar_frame, 
                            text="提示: 选择视频源后，点击开始追踪并在视频上框选目标物体")
        info_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # 初始化YOLO检测器
        if self.yolo_detector:
            yolo_detector = self.yolo_detector
            model_status_label.config(text="模型状态: 已加载", foreground="green")
        
        # 设置窗口最小大小
        yolo_window.update()
        yolo_window.minsize(yolo_window.winfo_width(), yolo_window.winfo_height())
        
        # 将窗口置于前台
        yolo_window.lift()
        yolo_window.focus_force()

    def toggle_yolo_tracking(self):
        """切换YOLO追踪状态"""
        if not self.yolo_available:
            messagebox.showwarning("功能不可用", "YOLO追踪功能不可用，请确保已安装必要的依赖")
            return
            
        # 切换状态
        self.using_yolo_tracker = not self.using_yolo_tracker
        
        if self.using_yolo_tracker:
            # 启用YOLO追踪
            self.tracking_algo_var.set("YOLO")
            self.yolo_toggle_btn.config(text="禁用YOLO追踪")
            self.yolo_status_label.config(text="已启用")
            
            # 更新VideoProcessor配置
            self.video_processor.config["tracking_algorithm"] = "YOLO"
            self.video_processor.config["yolo_model"] = self.yolo_model_var.get()
            
            messagebox.showinfo("YOLO追踪", "已启用YOLO追踪，后续视频分析将使用YOLO跟踪器")
        else:
            # 禁用YOLO追踪，回退到基础或智能跟踪器
            available_trackers = tracking_utils.get_available_algorithms()
            fallback = "智能" if "智能" in available_trackers else "Basic"
            self.tracking_algo_var.set(fallback)
            self.yolo_toggle_btn.config(text="启用YOLO追踪")
            self.yolo_status_label.config(text="未启动")
            
            # 更新VideoProcessor配置
            self.video_processor.config["tracking_algorithm"] = fallback
            
            messagebox.showinfo("YOLO追踪", f"已禁用YOLO追踪，后续视频分析将使用{fallback}跟踪器")
        
        # 重置当前的跟踪器
        if hasattr(self.video_processor, 'tracker'):
            self.video_processor.tracker = None
            logger.info("已重置当前跟踪器")
            
    def on_yolo_model_change(self, event=None):
        """处理YOLO模型变更"""
        if not self.yolo_available:
            return
            
        new_model = self.yolo_model_var.get()
        logger.info(f"YOLO模型变更为: {new_model}")
        
        # 如果当前正在使用YOLO跟踪器，需要更新配置
        if self.using_yolo_tracker:
            # 更新VideoProcessor配置
            self.video_processor.config["yolo_model"] = new_model
            
            # 重新初始化YOLO检测器
            try:
                from tracking_utils import init_yolo_detector
                self.yolo_detector = init_yolo_detector(new_model)
                if self.yolo_detector.initialized:
                    logger.info(f"YOLO检测器已更新为: {new_model}")
                    self.video_processor.config["yolo_detector"] = self.yolo_detector
                else:
                    logger.warning(f"更新YOLO检测器失败: {new_model}")
            except Exception as e:
                logger.error(f"更新YOLO检测器时出错: {e}")
            
            # 重置当前的跟踪器
            if hasattr(self.video_processor, 'tracker'):
                self.video_processor.tracker = None
                
            messagebox.showinfo("模型更新", f"已更新YOLO模型为 {new_model}，将在下次跟踪时生效")

    def toggle_camera_tracking(self):
        """切换摄像头追踪"""
        if not self.yolo_available:
            messagebox.showwarning("功能不可用", "YOLO 追踪功能不可用，请确保已安装必要的依赖")
            return
            
        if self.yolo_tracking_active:
            self.stop_tracking()
        else:
            self.start_camera_tracking()

    def open_video_for_tracking(self):
        """打开视频文件进行追踪"""
        if not self.yolo_available:
            messagebox.showwarning("功能不可用", "YOLO 追踪功能不可用，请确保已安装必要的依赖")
            return
            
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        self.video_path = file_path
        self.start_tracking(file_path)

    def initialize_yolo_detector(self):
        """初始化 YOLO 检测器"""
        try:
            from tracking_utils import init_yolo_detector
            
            logger.info(f"正在初始化 YOLO 检测器，模型: {self.yolo_model_name}")
            self.yolo_detector = init_yolo_detector(self.yolo_model_name)
            
            if self.yolo_detector.initialized:
                logger.info("YOLO 检测器初始化成功")
                return True
            else:
                logger.error("YOLO 检测器初始化失败")
                return False
        except Exception as e:
            logger.error(f"初始化 YOLO 检测器时出错: {e}")
            messagebox.showerror("初始化错误", f"初始化 YOLO 检测器时出错: {e}")
            return False

    def create_yolo_tracker(self):
        """创建 YOLO 追踪器"""
        try:
            from tracking_utils import create_yolo_tracker
            
            logger.info("正在创建 YOLO 追踪器")
            self.yolo_tracker = create_yolo_tracker(yolo_model=self.yolo_model_name)
            
            if self.yolo_tracker:
                logger.info("YOLO 追踪器创建成功")
                return True
            else:
                logger.error("YOLO 追踪器创建失败")
                return False
        except Exception as e:
            logger.error(f"创建 YOLO 追踪器时出错: {e}")
            messagebox.showerror("创建错误", f"创建 YOLO 追踪器时出错: {e}")
            return False

    def start_tracking(self, video_path=None):
        """启动 YOLO 视频追踪"""
        # 停止当前可能正在进行的追踪
        if self.yolo_tracking_active:
            self.stop_tracking()
        
        if video_path is None and hasattr(self, 'video_path'):
            video_path = self.video_path
        
        if not video_path:
            messagebox.showwarning("无视频源", "请先选择一个视频文件")
            return
        
        # 初始化 YOLO 检测器和追踪器
        if not self.initialize_yolo_detector():
            return
            
        if not self.create_yolo_tracker():
            return
        
        # 重置停止标志
        self.stop_tracking_event.clear()
        
        # 启动追踪线程
        self.tracking_thread = threading.Thread(
            target=self.run_video_tracking,
            args=(video_path,),
            daemon=True
        )
        self.tracking_thread.start()
        
        # 启动结果处理线程
        self.result_thread = threading.Thread(
            target=self.process_tracking_results,
            daemon=True
        )
        self.result_thread.start()
        
        # 更新UI
        self.yolo_tracking_active = True
        self.tracking_btn.config(text="停止追踪")
        self.tracking_status_label.config(text="正在追踪")

    def start_camera_tracking(self):
        """启动摄像头追踪"""
        # 停止当前可能正在进行的追踪
        if self.yolo_tracking_active:
            self.stop_tracking()
        
        # 初始化 YOLO 检测器和追踪器
        if not self.initialize_yolo_detector():
            return
            
        if not self.create_yolo_tracker():
            return
        
        # 重置停止标志
        self.stop_tracking_event.clear()
        
        # 启动摄像头追踪线程
        self.tracking_thread = threading.Thread(
            target=self.run_camera_tracking,
            daemon=True
        )
        self.tracking_thread.start()
        
        # 启动结果处理线程
        self.result_thread = threading.Thread(
            target=self.process_tracking_results,
            daemon=True
        )
        self.result_thread.start()
        
        # 更新UI
        self.yolo_tracking_active = True
        self.camera_btn.config(text="关闭摄像头")
        self.tracking_status_label.config(text="摄像头追踪中")

    def stop_tracking(self):
        """停止 YOLO 追踪"""
        if not self.yolo_tracking_active:
            return
            
        # 设置停止事件
        self.stop_tracking_event.set()
        
        # 等待线程结束
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=2.0)
        
        # 重置状态
        self.yolo_tracking_active = False
        self.tracking_btn.config(text="开始追踪")
        self.camera_btn.config(text="打开摄像头")
        self.tracking_status_label.config(text="已停止")

    def run_video_tracking(self, video_path):
        """视频追踪处理主循环"""
        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return
                
            # 读取第一帧
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取视频帧")
                return
                
            # 调整帧大小以适应显示
            frame = self.resize_frame_for_display(frame)
                
            # 选择目标区域
            roi = self.select_roi(frame)
            if roi is None or roi[2] <= 0 or roi[3] <= 0:
                logger.info("未选择目标区域，取消追踪")
                return
                
            # 初始化追踪器
            success = self.yolo_tracker.init(frame, roi)
            if not success:
                logger.error("初始化追踪器失败")
                return
                
            # 追踪循环
            while not self.stop_tracking_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 调整帧大小
                frame = self.resize_frame_for_display(frame)
                    
                # 更新追踪
                success, box = self.yolo_tracker.update(frame)
                
                if success:
                    # 绘制结果
                    result_frame = self.yolo_tracker.draw_bbox(frame)
                    
                    # 添加到队列
                    if not self.tracking_results_queue.full():
                        self.tracking_results_queue.put(result_frame)
                else:
                    # 追踪失败
                    if not self.tracking_results_queue.full():
                        cv2.putText(frame, "追踪失败", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.tracking_results_queue.put(frame)
                        
                # 控制处理速度
                time.sleep(0.03)  # ~30fps
                
            # 关闭视频
            cap.release()
            
        except Exception as e:
            logger.error(f"视频追踪过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保状态被重置
            self.root.after(0, self.reset_tracking_ui)

    def run_camera_tracking(self):
        """摄像头追踪处理主循环"""
        try:
            # 打开摄像头
            cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
            if not cap.isOpened():
                logger.error("无法打开摄像头")
                return
                
            # 读取第一帧
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头帧")
                return
                
            # 调整帧大小以适应显示
            frame = self.resize_frame_for_display(frame)
                
            # 选择目标区域
            roi = self.select_roi(frame)
            if roi is None or roi[2] <= 0 or roi[3] <= 0:
                logger.info("未选择目标区域，取消追踪")
                return
                
            # 初始化追踪器
            success = self.yolo_tracker.init(frame, roi)
            if not success:
                logger.error("初始化追踪器失败")
                return
                
            # 追踪循环
            while not self.stop_tracking_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 调整帧大小
                frame = self.resize_frame_for_display(frame)
                    
                # 更新追踪
                success, box = self.yolo_tracker.update(frame)
                
                if success:
                    # 绘制结果
                    result_frame = self.yolo_tracker.draw_bbox(frame)
                    
                    # 添加到队列
                    if not self.tracking_results_queue.full():
                        self.tracking_results_queue.put(result_frame)
                else:
                    # 追踪失败
                    if not self.tracking_results_queue.full():
                        cv2.putText(frame, "追踪失败", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.tracking_results_queue.put(frame)
                        
                # 控制处理速度
                time.sleep(0.03)  # ~30fps
                
            # 关闭摄像头
            cap.release()
            
        except Exception as e:
            logger.error(f"摄像头追踪过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保状态被重置
            self.root.after(0, self.reset_tracking_ui)

    def process_tracking_results(self):
        """处理追踪结果队列"""
        try:
            while not self.stop_tracking_event.is_set() or not self.tracking_results_queue.empty():
                # 从队列中获取结果帧
                try:
                    result_frame = self.tracking_results_queue.get(timeout=0.5)
                    
                    # 转换为 PIL 图像并显示
                    pil_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                    self.display_pil_image(pil_img)
                    
                    # 重要：标记任务完成
                    self.tracking_results_queue.task_done()
                except Exception as e:
                    # 超时或其他错误，继续循环
                    continue
        except Exception as e:
            logger.error(f"处理追踪结果时出错: {e}")
            
    def reset_tracking_ui(self):
        """重置追踪UI状态"""
        self.yolo_tracking_active = False
        self.tracking_btn.config(text="开始追踪")
        self.camera_btn.config(text="打开摄像头")
        self.tracking_status_label.config(text="已停止")

    def resize_frame_for_display(self, frame, max_width=800, max_height=600):
        """调整帧大小以适应显示"""
        height, width = frame.shape[:2]
        
        # 计算缩放比例
        scale = min(max_width / width, max_height / height)
        
        # 如果小于最大尺寸，无需缩放
        if width <= max_width and height <= max_height:
            return frame
            
        # 计算新尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 调整大小
        resized = cv2.resize(frame, (new_width, new_height))
        return resized

    def select_roi(self, frame):
        """选择追踪目标区域"""
        # 创建临时窗口
        window_name = "选择追踪目标 (拖动鼠标选择，按Enter确认)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        
        # 使用OpenCV的selectROI功能
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        
        # 关闭临时窗口
        cv2.destroyWindow(window_name)
        
        return roi

    def display_pil_image(self, pil_img):
        """在主界面上显示PIL图像"""
        # 调整图像大小以适应显示区域
        if hasattr(self, 'preview_canvas'):
            # 获取画布尺寸
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # 如果尺寸有效，调整图像大小
            if canvas_width > 1 and canvas_height > 1:
                pil_img = self.resize_pil_image(pil_img, canvas_width, canvas_height)
            
            # 转换为PhotoImage并显示
            self.current_image = ImageTk.PhotoImage(pil_img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        
    def resize_pil_image(self, img, max_width, max_height):
        """调整PIL图像大小以适应显示区域，保持宽高比"""
        width, height = img.size
        
        # 计算缩放比例
        scale = min(max_width / width, max_height / height)
        
        # 如果小于最大尺寸，无需缩放
        if width <= max_width and height <= max_height:
            return img
            
        # 计算新尺寸
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 调整大小
        return img.resize((new_width, new_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)

    def start_ui_update_thread(self):
        """Start a thread to update UI with video processing results"""
        def update_ui():
            while True:
                if self.video_processor and self.video_processor.is_processing:
                    result = self.video_processor.get_next_result()
                    if result and self.visualization_panel:
                        # Update visualization panel with the new frame
                        mosaic = result["visualizations"].get("mosaic")
                        if mosaic is not None:
                            self.visualization_panel.add_video_frame(
                                mosaic,
                                timestamp=result.get("timestamp"),
                                metadata={
                                    "tracks": result.get("tracks", {}),
                                    "detections": result.get("detections", [])
                                }
                            )
                time.sleep(0.05)  # Short sleep to prevent high CPU usage
                
        thread = threading.Thread(target=update_ui, daemon=True)
        thread.start()
        
    def toggle_ui_state(self, state):
        """Enable or disable UI elements during processing operations"""
        # Disable/enable all buttons and interactive elements
        for widget in self.root.winfo_children():
            if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Scale)):
                widget.config(state=state)
            elif isinstance(widget, tk.Text) and widget != self.output_text:
                widget.config(state=state)
            elif hasattr(widget, 'winfo_children'):
                for child in widget.winfo_children():
                    if isinstance(child, (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Scale)):
                        child.config(state=state)
                    elif isinstance(child, tk.Text) and child != self.output_text:
                        child.config(state=state)
                    elif hasattr(child, 'winfo_children'):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Scale)):
                                grandchild.config(state=state)
                            elif isinstance(grandchild, tk.Text) and grandchild != self.output_text:
                                grandchild.config(state=state)
        
        # Always keep the output text in a state where it can be updated
        if self.output_text:
            self.output_text.config(state=tk.NORMAL)
        
        # Update UI to reflect changes
        self.root.update()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="新建会话", command=self.new_session)
        file_menu.add_command(label="保存会话", command=self.save_session)
        file_menu.add_command(label="导出CSV", command=self.export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="编辑", menu=edit_menu)
        edit_menu.add_command(label="清除媒体", command=self.clear_media)
        edit_menu.add_command(label="清除对话", command=self.clear_conversation)
        edit_menu.add_command(label="清除可视化", command=self.clear_visualizations)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视图", menu=view_menu)
        view_menu.add_checkbutton(label="批处理模式", variable=self.batch_mode)
        
        # Video menu
        video_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="视频", menu=video_menu)
        video_menu.add_command(label="播放", command=self.play_video)
        video_menu.add_command(label="暂停", command=self.pause_video)
        video_menu.add_command(label="停止", command=self.stop_video)
        video_menu.add_separator()
        video_menu.add_command(label="启动目标跟踪", command=self.start_video_tracking)
        video_menu.add_command(label="停止目标跟踪", command=self.stop_video_tracking)
        video_menu.add_command(label="导出跟踪结果", command=self.export_tracking_results)
        
        # Export menu
        export_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="导出", menu=export_menu)
        export_menu.add_command(label="导出可视化", command=self.export_visualizations)
        export_menu.add_command(label="导出分析结果", command=self.export_results)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)

    def new_session(self):
        if messagebox.askyesno("新建会话", "是否保存当前会话？"):
            self.save_session()
        self.history_manager = HistoryManager()
        self.conversation = []
        self.media_items = []
        self.preview_canvas.items = []
        self.preview_canvas.update_previews()
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)
        if self.visualization_panel:
            self.visualization_panel.clear_all()
        
        # Reset video processor
        self.video_processor.reset()

    def save_session(self):
        self.history_manager.add_entry({
            "conversation": self.conversation,
            "media_items": self.media_items
        })
        self.history_manager.save_session()
        messagebox.showinfo("保存成功", "会话已保存")

    def export_csv(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.history_manager.export_csv(file_path)
            messagebox.showinfo("导出成功", f"会话已导出到: {file_path}")

    def clear_media(self):
        if messagebox.askyesno("清除媒体", "确定要清除所有媒体文件吗？"):
            # Stop any video processing first
            self.stop_video_tracking()
            self.stop_video()
            
            self.media_items = []
            self.preview_canvas.items = []
            self.preview_canvas.update_previews()

    def clear_conversation(self):
        if messagebox.askyesno("清除对话", "确定要清除所有对话记录吗？"):
            self.conversation = []
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.config(state=tk.DISABLED)
    
    def clear_visualizations(self):
        if self.visualization_panel:
            self.visualization_panel.clear_all()
    
    def export_visualizations(self):
        if self.visualization_panel:
            self.visualization_panel.export_all()
        else:
            messagebox.showinfo("提示", "无可视化面板")

    def show_help(self):
        help_text = """
智能多模态分析系统 v4.0 使用说明

1. 基本功能
   - 支持图片、视频文件上传和URL添加
   - 多种AI模型可选
   - 目标检测和场景分析
   - 批量处理功能
   - 高级可视化和标注功能

2. 视频跟踪功能
   - 支持上传本地视频文件
   - 实时目标跟踪
   - 多种跟踪算法可选
   - 轨迹可视化
   - 跟踪结果导出

3. 快捷键
   - Ctrl+N: 新建会话
   - Ctrl+S: 保存会话
   - Ctrl+E: 导出CSV
   - Ctrl+Q: 退出
   - 空格键: 播放/暂停视频

4. 注意事项
   - 请确保设置了正确的API密钥
   - 大文件处理可能需要较长时间
   - 建议定期保存会话
   
5. 可视化功能
   - 目标检测结果自动生成标注可视化
   - 可导出单个或批量可视化结果
   - 支持多种标注颜色和样式
        """
        help_window = tk.Toplevel(self.root)
        help_window.title("使用说明")
        help_window.geometry("600x500")
        
        text = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)

    def show_about(self):
        about_text = """
智能多模态分析系统 v4.0

- 支持多种AI模型
- 先进的目标检测
- 视频实时跟踪
- 批量处理能力
- 历史记录管理
- 高级可视化和标注

Copyright © 2025
        """
        messagebox.showinfo("关于", about_text)

    def style_widgets(self):
        style = ttk.Style()
        style.configure("TFrame", background="#f8f9fa")
        style.configure("TButton", padding=6, relief="flat", background="#4CAF50", font=("TkDefaultFont", 12))
        style.configure("TLabel", background="#f8f9fa", font=("TkDefaultFont", 12, "bold"))
        style.configure("Accent.TButton", background="#2196F3")
        style.map("TButton",
                 background=[("active", "#45a049"), ("disabled", "#cccccc")])


    def create_ui(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_frame = ttk.Frame(main_pane, width=500)
        main_pane.add(left_frame)
        
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame)
        
        self.create_media_panel(left_frame)
        self.create_control_panel(left_frame)
        self.create_output_panel(right_frame)
        self.create_chat_panel(right_frame)
        self.create_prompt_panel(right_frame)
        self.create_status_bar()

    def create_media_panel(self, parent):
        media_frame = ttk.LabelFrame(parent, text="媒体工作区", padding=10)
        media_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        toolbar = ttk.Frame(media_frame)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(toolbar, text="重置缩放", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="适应窗口", command=self.fit_to_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="删除选中", command=self.delete_selected).pack(side=tk.LEFT, padx=2)
        
        self.preview_canvas = MediaPreviewCanvas(media_frame)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar_frame = ttk.Frame(media_frame)
        scrollbar_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        hbar = ttk.Scrollbar(media_frame, orient=tk.HORIZONTAL, command=self.preview_canvas.xview)
        vbar = ttk.Scrollbar(scrollbar_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        self.preview_canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="操作控制台", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # File operations
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="上传文件", command=self.upload_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="上传视频", command=self.upload_video).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="批量导入", command=self.batch_import).pack(side=tk.LEFT, padx=2)
        
        # URL input
        url_frame = ttk.Frame(control_frame)
        url_frame.pack(fill=tk.X, pady=5)
        
        self.url_entry = ttk.Entry(url_frame)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(url_frame, text="添加URL", command=self.add_url).pack(side=tk.RIGHT)
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="模型:").pack(side=tk.LEFT)
        model_menu = ttk.Combobox(model_frame, textvariable=self.model_var,
                                values=list(SUPPORTED_MODELS.keys()))
        model_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        model_menu.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # Analysis buttons
        analysis_frame = ttk.Frame(control_frame)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="目标定位", 
                  command=lambda: self.run_detection()).pack(side=tk.LEFT, padx=2)
        ttk.Button(analysis_frame, text="场景分析",
                  command=lambda: self.analyze_scene()).pack(side=tk.LEFT, padx=2)
        ttk.Button(analysis_frame, text="文本提取",
                  command=lambda: self.extract_text()).pack(side=tk.LEFT, padx=2)
                  
        # Video control buttons
        video_frame = ttk.LabelFrame(control_frame, text="视频控制", padding=5)
        video_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(video_frame, text="播放", 
                  command=self.play_video).pack(side=tk.LEFT, padx=2)
        ttk.Button(video_frame, text="暂停", 
                  command=self.pause_video).pack(side=tk.LEFT, padx=2)
        ttk.Button(video_frame, text="停止", 
                  command=self.stop_video).pack(side=tk.LEFT, padx=2)
                  
        
        # Tracking algorithm selection
        ttk.Label(video_frame, text="跟踪算法:").pack(side=tk.LEFT, padx=(10, 0))
        
        # Get available trackers
        available_trackers = tracking_utils.get_available_algorithms()
        
        tracking_combo = ttk.Combobox(
            video_frame, textvariable=self.tracking_algo_var,
            values=available_trackers,
            width=10
        )
        tracking_combo.pack(side=tk.LEFT, padx=5)
        
        # Advanced options
        advanced_frame = ttk.LabelFrame(control_frame, text="高级选项", padding=5)
        advanced_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(advanced_frame, text="批处理模式",
                       variable=self.batch_mode).pack(side=tk.LEFT)
        
        ttk.Label(advanced_frame, text="温度:").pack(side=tk.LEFT, padx=(10, 0))
        temperature_scale = ttk.Scale(advanced_frame, from_=0.0, to=1.0,
                                    variable=self.temperature_var, orient=tk.HORIZONTAL)
        temperature_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Visualization options
        vis_frame = ttk.LabelFrame(control_frame, text="可视化选项", padding=5)
        vis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(vis_frame, text="显示标签", 
                       variable=self.show_labels_var).pack(side=tk.LEFT)
        
        ttk.Checkbutton(vis_frame, text="显示置信度", 
                       variable=self.show_scores_var).pack(side=tk.LEFT, padx=(10, 0))
        
        # Color scheme selection
        ttk.Label(vis_frame, text="颜色方案:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(vis_frame, textvariable=self.color_scheme_var, 
                    values=["default", "rainbow", "pastel", "highvis"]).pack(side=tk.LEFT)
        
        # Video tracking buttons
        track_frame = ttk.Frame(control_frame)
        track_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(track_frame, text="启动跟踪", 
                  command=self.start_video_tracking).pack(side=tk.LEFT, padx=2)
        ttk.Button(track_frame, text="停止跟踪", 
                  command=self.stop_video_tracking).pack(side=tk.LEFT, padx=2)
        ttk.Button(track_frame, text="导出轨迹", 
                  command=self.export_tracking_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(track_frame, text="可视化轨迹", 
                  command=self.visualize_trajectories).pack(side=tk.LEFT, padx=2)




    def create_output_panel(self, parent):
        self.output_notebook = CustomNotebook(parent)
        self.output_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Main output tab
        output_frame = ttk.Frame(self.output_notebook)
        self.output_notebook.add(output_frame, text="分析结果")
        
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, state=tk.DISABLED,
                                 bg="white", font=("TkDefaultFont", 12))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        
        # History tab
        history_frame = ttk.Frame(self.output_notebook)
        self.output_notebook.add(history_frame, text="历史记录")
        
        self.history_text = tk.Text(history_frame, wrap=tk.WORD, state=tk.DISABLED,
                                  bg="white", font=("TkDefaultFont", 12))
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        history_scrollbar = ttk.Scrollbar(history_frame, command=self.history_text.yview)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.config(yscrollcommand=history_scrollbar.set)
        
        # Visualization tab
        vis_frame = ttk.Frame(self.output_notebook)
        self.output_notebook.add(vis_frame, text="目标可视化")
        
        self.visualization_panel = VisualizationPanel(vis_frame)
        self.visualization_panel.pack(fill=tk.BOTH, expand=True)
        
        # Video analysis tab
        video_frame = ttk.Frame(self.output_notebook)
        self.output_notebook.add(video_frame, text="视频分析")
        
        self.video_notebook = ttk.Notebook(video_frame)
        self.video_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Video tabs
        video_view_frame = ttk.Frame(self.video_notebook)
        self.video_notebook.add(video_view_frame, text="视频查看器")
        
        self.video_canvas = tk.Canvas(video_view_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        video_controls_frame = ttk.Frame(video_view_frame)
        video_controls_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.video_slider = ttk.Scale(video_controls_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.video_slider.pack(fill=tk.X, side=tk.TOP, padx=5, pady=5)
        
        # Tracking tab
        tracking_frame = ttk.Frame(self.video_notebook)
        self.video_notebook.add(tracking_frame, text="目标跟踪")
        
        self.tracking_canvas = tk.Canvas(tracking_frame, bg="black")
        self.tracking_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Analysis results tab
        analysis_frame = ttk.Frame(self.video_notebook)
        self.video_notebook.add(analysis_frame, text="分析结果")
        
        self.video_analysis_text = tk.Text(analysis_frame, wrap=tk.WORD, state=tk.DISABLED,
                                         bg="white", font=("TkDefaultFont", 12))
        self.video_analysis_text.pack(fill=tk.BOTH, expand=True)

    def create_chat_panel(self, parent):
        chat_frame = ttk.LabelFrame(parent, text="智能对话", padding=10)
        chat_frame.pack(fill=tk.X, pady=5)
        
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, expand=True)
        
        self.chat_entry = ttk.Entry(input_frame, font=("TkDefaultFont", 12))
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.chat_entry.bind("<Return>", lambda e: self.send_message())
        
        ttk.Button(input_frame, text="发送", 
                  command=self.send_message).pack(side=tk.RIGHT)

    def create_prompt_panel(self, parent):
        prompt_frame = ttk.LabelFrame(parent, text="提示词模板", padding=10)
        prompt_frame.pack(fill=tk.X, pady=5)
        
        # Template selection
        template_frame = ttk.Frame(prompt_frame)
        template_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(template_frame, text="预设模板:").pack(side=tk.LEFT)
        self.template_var = tk.StringVar()
        template_menu = ttk.Combobox(template_frame, textvariable=self.template_var,
                                values=list(PROMPT_TEMPLATES.keys()))
        template_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        template_menu.bind("<<ComboboxSelected>>", self.on_template_select)
        
        # Custom prompt
        self.prompt_text = tk.Text(prompt_frame, height=4, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(prompt_frame)
        button_frame.pack(fill=tk.X)
        
        # 添加视频分析按钮 - 打开独立窗口
        ttk.Button(
            button_frame, 
            text="视频分析",
            command=self.open_video_analysis_window
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(button_frame, text="应用",
                command=self.apply_prompt).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="保存",
                command=self.save_prompt).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="重置",
                command=self.reset_prompt).pack(side=tk.LEFT, padx=2)

    def open_video_analysis_window(self):
        """打开视频内容分析窗口，使用指定的大模型"""
        try:
            # 动态导入所需模块
            from video_analysis_window import VideoAnalysisWindow
            from model_api import ModelAPIClient
            import os
            import logging
            
            # 创建API客户端，使用正确的模型名称
            
            api_client = ModelAPIClient(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model_name="qwen-vl-max"  # 使用API支持的模型名称
            )
            
            # 获取主窗口
            root_window = self.root
            
            # 创建新窗口而不是在当前窗口打开
            analysis_window = tk.Toplevel(root_window)
            analysis_window.title("视频内容分析")
            analysis_window.geometry("1200x800")
            analysis_window.minsize(800, 600)  # 设置最小尺寸
            
            # 确保窗口被完全创建
            analysis_window.update()
            
            # 创建视频分析界面
            video_window = VideoAnalysisWindow(
                analysis_window, 
                return_callback=lambda: analysis_window.destroy(),
                api_caller=api_client.analyze_video_frames
            )
            
            # 将窗口对象保存在类变量中，防止被垃圾回收
            self.video_analysis_window = analysis_window
            self.video_analysis_ui = video_window
            
            logging.info("成功打开视频分析窗口")
            
        except ImportError as e:
            from tkinter import messagebox
            messagebox.showerror("导入错误", f"无法加载视频分析模块: {str(e)}\n请确保已安装所有必要的依赖。")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("错误", f"打开视频分析窗口时出错: {str(e)}")

    def create_status_bar(self):
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.status_bar, text="就绪")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(self.status_bar, mode="indeterminate")
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # Add current time display
        self.time_label = ttk.Label(self.status_bar, text="")
        self.time_label.pack(side=tk.RIGHT, padx=10)
        self.update_clock()
        
    def update_clock(self):
        """Update the clock display in status bar"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.configure(text=current_time)
        self.root.after(1000, self.update_clock)  # Schedule update every 1 second
        
    # Media handling methods
    def reset_zoom(self):
        self.preview_canvas.scale = 1.0
        self.preview_canvas.update_previews()

    def fit_to_window(self):
        if not self.media_items:
            return
        
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        # Calculate the optimal scale
        max_width = max(item.get("width", 0) for item in self.media_items)
        max_height = max(item.get("height", 0) for item in self.media_items)
        
        if max_width and max_height:
            width_scale = (canvas_width - 20) / max_width
            height_scale = (canvas_height - 20) / max_height
            self.preview_canvas.scale = min(width_scale, height_scale)
            self.preview_canvas.update_previews()

    def delete_selected(self):
        selected_item = self.preview_canvas.get_selected_item()
        if selected_item and messagebox.askyesno("删除确认", "是否删除选中的媒体项？"):
            # If it's a video that's currently playing, stop playback
            if selected_item.get("type") == "video" and self.is_playing_video:
                self.stop_video()
                
            self.media_items.remove(selected_item)
            self.preview_canvas.items = self.media_items
            self.preview_canvas.selected_item = None
            self.preview_canvas.update_previews()
    
    def upload_files(self):
        """Handle file upload functionality"""
        file_paths = filedialog.askopenfilenames(
            title="选择媒体文件",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            return
            
        for file_path in file_paths:
            self.add_media(file_path, True)
            
    def upload_video(self):
        """Handle video file upload functionality"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        self.add_media(file_path, True, force_video=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_video_response(self, url):
        """带重试机制的视频资源获取"""
        response = requests.get(url, timeout=(10, 30), stream=True)
        response.raise_for_status()
        return response

    def add_media(self, source, is_local=True, force_video=False):
        """Add media item to the preview canvas"""
        try:
            media_type = "video" if force_video else self.get_media_type(source)
            if not media_type:
                return
                
            if media_type == "image":
                if is_local:
                    img = Image.open(source)
                else:
                    response = requests.get(source)
                    img = Image.open(BytesIO(response.content))
                
                # Resize image for preview
                img.thumbnail((160, 160), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                item = {
                    "type": "image",
                    "source": source,
                    "preview": photo,
                    "is_local": is_local,
                    "width": img.width,
                    "height": img.height,
                    "original": img.copy()  # Store original image for visualization
                }
                
            elif media_type == "video":
                # 视频处理：获取第一帧作为预览
                if not is_local:
                    response = self.get_video_response(source)
                    temp_file = Path("temp_video.mp4")
                    with open(temp_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    source = str(temp_file)
                    is_local = True
                
                # Open video file
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    raise ValueError(f"无法打开视频文件: {source}")
                    
                # Get video metadata
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Get first frame for preview
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("无法读取视频帧")
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((160, 160), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                item = {
                    "type": "video",
                    "source": source,
                    "preview": photo,
                    "is_local": is_local,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": frame_count,
                    "thumbnail": img.copy()
                }
                
                cap.release()
                
            self.media_items.append(item)
            self.preview_canvas.items = self.media_items
            self.preview_canvas.update_previews()
            
        except Exception as e:
            logger.error(f"Error adding media: {str(e)}")
            messagebox.showerror("错误", f"添加媒体失败: {str(e)}")

    def get_media_type(self, source):
        image_ext = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif')
        video_ext = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg', '.webm')
        
        lower_source = source.lower()
        if any(lower_source.endswith(ext) for ext in image_ext):
            return "image"
        elif any(lower_source.endswith(ext) for ext in video_ext):
            return "video"
        else:
            return None
    
    def add_url(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("警告", "请输入URL")
            return

        try:
            # 检查URL结构
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not all([parsed.scheme in ['http', 'https'], parsed.netloc]):
                raise ValueError("无效的URL结构")

            # 获取Content-Type
            response = requests.head(url, allow_redirects=True)
            content_type = response.headers.get('Content-Type', '')

            # 根据Content-Type判断媒体类型
            media_type = None
            if 'image' in content_type:
                media_type = 'image'
            elif 'video' in content_type:
                media_type = 'video'
            else:
                # 回退到扩展名检查
                media_type = self.get_media_type(url)

            if not media_type:
                raise ValueError("不支持的文件类型")

            self.add_media(url, is_local=False, force_video=(media_type=='video'))
            self.url_entry.delete(0, tk.END)

        except Exception as e:
            messagebox.showerror("错误", f"无法处理URL: {str(e)}")

    def batch_import(self):
        """批量导入媒体文件"""
        folder = filedialog.askdirectory(title="选择媒体文件夹")
        if not folder:
            return
            
        extensions = (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov")
        files = []
        for ext in extensions:
            files.extend(Path(folder).glob(f"*{ext}"))
            
        if not files:
            messagebox.showinfo("提示", "未找到支持的媒体文件")
            return
            
        progress = tk.Toplevel(self.root)
        progress.title("批量导入进度")
        progress.geometry("300x150")
        
        label = ttk.Label(progress, text="正在导入...")
        label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress, length=200, mode="determinate")
        progress_bar.pack(pady=10)
        
        progress_bar["maximum"] = len(files)
        for i, file in enumerate(files, 1):
            force_video = self.get_media_type(str(file)) == "video"
            self.add_media(str(file), True, force_video=force_video)
            progress_bar["value"] = i
            label.config(text=f"正在导入 ({i}/{len(files)})")
            progress.update()
            
        progress.destroy()
        messagebox.showinfo("完成", f"成功导入 {len(files)} 个文件")

    def on_model_change(self, event=None):
        """模型切换处理"""
        model = self.model_var.get()
        model_config = SUPPORTED_MODELS[model]
        
        # 更新温度滑块
        self.temperature_var.set(model_config.get("temperature", 0.7))
        
        # 更新视频处理器的模型设置
        if self.video_processor:
            self.video_processor.model_name = model
        
        # 检查媒体兼容性
        supported = model_config["supports"]
        incompatible = [item for item in self.media_items 
                       if item["type"] not in supported]
        
        if incompatible:
            messagebox.showwarning(
                "兼容性警告",
                f"当前模型不支持以下类型: {', '.join(set(item['type'] for item in incompatible))}"
            )

    def on_template_select(self, event=None):
        """提示词模板选择处理"""
        template_name = self.template_var.get()
        if template_name in PROMPT_TEMPLATES:
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(tk.END, PROMPT_TEMPLATES[template_name])
            
    def save_prompt(self):
        """保存自定义提示词模板"""
        name = simpledialog.askstring("保存模板", "请输入模板名称:")
        if name:
            PROMPT_TEMPLATES[name] = self.prompt_text.get(1.0, tk.END).strip()
            messagebox.showinfo("成功", "模板已保存")

    def reset_prompt(self):
        """重置提示词"""
        self.prompt_text.delete(1.0, tk.END)
        self.template_var.set("")

    def apply_prompt(self):
        """应用当前提示词"""
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("警告", "提示词不能为空")
            return
            
        try:
            # 验证提示词格式
            if not prompt.startswith("<") or not prompt.endswith(">"):
                messagebox.showwarning("警告", "提示词格式不正确，请使用<>包裹指令")
                return
                
            # 保存为当前会话的提示词模板
            self.prompt_template = prompt
            messagebox.showinfo("成功", "提示词已应用")
        except Exception as e:
            logger.error(f"Apply prompt error: {str(e)}")
            messagebox.showerror("错误", f"应用提示词失败: {str(e)}")
            
    # Video playback and tracking methods
    def play_video(self):
        """Play selected video"""
        selected_item = self.preview_canvas.get_selected_item()
        if not selected_item or selected_item["type"] != "video":
            messagebox.showinfo("提示", "请先选择一个视频文件")
            return
            
        if self.is_playing_video:
            return  # Already playing
            
        # Switch to video analysis tab
        for i in range(self.output_notebook.index("end")):
            if "视频分析" in self.output_notebook.tab(i, "text"):
                self.output_notebook.select(i)
                break
        
        # Load video into processor if needed
        if not self.video_processor.current_video:
            success = self.video_processor.load_video(selected_item["source"])
            if not success:
                messagebox.showerror("错误", "无法加载视频文件")
                return
        
        # Start playback in a thread
        self.is_playing_video = True
        self.video_play_thread = threading.Thread(target=self._play_video_thread, args=(selected_item,))
        self.video_play_thread.daemon = True
        self.video_play_thread.start()
    
    def _play_video_thread(self, video_item):
        """Thread for playing video"""
        try:
            # Open video file
            cap = cv2.VideoCapture(video_item["source"])
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_item['source']}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default if FPS can't be determined
                
            delay = 1.0 / fps
            
            # Update status
            self.status_label.config(text=f"正在播放视频: {Path(video_item['source']).name}")
            
            # Update slider max value
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_slider.config(to=total_frames)
            
            frame_count = 0
            while self.is_playing_video and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Update slider position
                frame_count += 1
                self.video_slider.set(frame_count)
                
                # Convert frame to display format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize to fit canvas
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has valid dimensions
                    img_width, img_height = img.size
                    # Calculate scaling to fit canvas
                    scale = min(canvas_width/img_width, canvas_height/img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image=img)
                
                # Update canvas
                self.video_canvas.delete("all")
                self.video_canvas.create_image(
                    canvas_width//2, canvas_height//2,
                    image=photo, anchor=tk.CENTER
                )
                self.video_canvas.image = photo  # Keep reference
                
                # Process GUI events
                self.root.update_idletasks()
                self.root.update()
                
                # Control frame rate
                time.sleep(delay)
                
            # Video finished or stopped
            cap.release()
            self.status_label.config(text="视频播放完成")
            self.is_playing_video = False
            
        except Exception as e:
            logger.error(f"Video playback error: {str(e)}")
            self.status_label.config(text=f"视频播放错误: {str(e)}")
            self.is_playing_video = False
    
    def pause_video(self):
        """Pause video playback"""
        if self.is_playing_video:
            self.is_playing_video = False
            self.status_label.config(text="视频已暂停")
    
    def stop_video(self):
        """Stop video playback"""
        self.is_playing_video = False
        if self.video_play_thread:
            self.video_play_thread.join(timeout=1.0)
            self.video_play_thread = None
        
        self.video_canvas.delete("all")
        self.status_label.config(text="视频已停止")
    
    def start_video_tracking(self):
        """Start video tracking with AI detection"""
        selected_item = self.preview_canvas.get_selected_item()
        if not selected_item or selected_item["type"] != "video":
            messagebox.showinfo("提示", "请先选择一个视频文件")
            return
            
        # Switch to video analysis tab
        for i in range(self.output_notebook.index("end")):
            if "视频分析" in self.output_notebook.tab(i, "text"):
                self.output_notebook.select(i)
                break
                
        # Stop any current tracking
        self.stop_video_tracking()
        
        # Update configuration
        config = {
            "model": self.model_var.get(),
            "temperature": self.temperature_var.get(),
            "tracking_algorithm": self.tracking_algo_var.get(),
            "detection_interval": 30,  # Run detection every 30 frames
            "visualization": {
                "show_labels": self.show_labels_var.get(),
                "show_scores": self.show_scores_var.get(),
                "color_scheme": self.color_scheme_var.get()
            }
        }
        
        # Update video processor
        self.video_processor.config = config
        self.video_processor.model_name = config["model"]
        
        # Load video and start processing
        success = self.video_processor.load_video(selected_item["source"])
        if not success:
            messagebox.showerror("错误", "无法加载视频文件")
            return
            
        # Start processing
        self.video_processor.start_processing()
        self.status_label.config(text=f"正在分析视频: {Path(selected_item['source']).name}")
    
    def stop_video_tracking(self):
        """Stop video tracking"""
        if self.video_processor:
            self.video_processor.stop_processing()
        self.status_label.config(text="视频分析已停止")
    
    def export_tracking_results(self):
        """Export tracking results to file"""
        if not self.video_processor or not self.video_processor.tracker.tracks:
            messagebox.showinfo("提示", "无跟踪结果可导出")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            result = self.video_processor.export_tracking_results(file_path)
            if result:
                messagebox.showinfo("成功", f"跟踪结果已导出至: {file_path}")
            else:
                messagebox.showerror("错误", "导出跟踪结果失败")
    
    def visualize_trajectories(self):
        """Visualize object trajectories"""
        if not self.video_processor or not self.video_processor.tracks.track_history:
            messagebox.showinfo("提示", "无轨迹数据可视化")
            return
            
        # Create trajectory visualization
        selected_item = self.preview_canvas.get_selected_item()
        if not selected_item or selected_item["type"] != "video":
            messagebox.showinfo("提示", "请先选择一个视频文件")
            return
            
        # Get first frame for background
        cap = cv2.VideoCapture(selected_item["source"])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("错误", "无法读取视频帧作为背景")
            return
            
        # Create visualization with trajectories
        vis_config = {
            "show_labels": self.show_labels_var.get(),
            "show_scores": self.show_scores_var.get(),
            "color_scheme": self.color_scheme_var.get()
        }
        
        track_vis = visualize_tracking(
            frame, 
            self.video_processor.tracks,
            self.video_processor.tracker.track_history,
            vis_config
        )
        
        # Add to visualization panel
        if self.visualization_panel:
            metadata = {
                "source": Path(selected_item["source"]).name,
                "tracks": len(self.video_processor.tracks),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.visualization_panel.add_visualization(track_vis, metadata)
            
            # Switch to visualization tab
            for i in range(self.output_notebook.index("end")):
                if "目标可视化" in self.output_notebook.tab(i, "text"):
                    self.output_notebook.select(i)
                    break

    def analyze_scene(self):
        """场景分析"""
        if not self.media_items:
            messagebox.showwarning("警告", "请先添加媒体文件")
            return
            
        model = self.model_var.get()
        if "image" not in SUPPORTED_MODELS[model]["supports"]:
            messagebox.showerror("错误", "当前模型不支持图像分析")
            return
            
        self.toggle_ui_state(tk.DISABLED)
        self.status_label.config(text="正在进行场景分析...")
        self.progress_bar.start()
        
        messages = [{
            "role": "system",
            "content": PROMPT_TEMPLATES["scene_analysis"]
        }]
        
        threading.Thread(target=self._run_analysis,
                       args=(messages, "scene_analysis")).start()

    def extract_text(self):
        """文本提取"""
        if not self.media_items:
            messagebox.showwarning("警告", "请先添加媒体文件")
            return
            
        model = self.model_var.get()
        if "image" not in SUPPORTED_MODELS[model]["supports"]:
            messagebox.showerror("错误", "当前模型不支持OCR")
            return
            
        self.toggle_ui_state(tk.DISABLED)
        self.status_label.config(text="正在提取文本...")
        self.progress_bar.start()
        
        messages = [{
            "role": "system",
            "content": PROMPT_TEMPLATES["text_extraction"]
        }]
        
        threading.Thread(target=self._run_analysis,
                       args=(messages, "text_extraction")).start()

    def run_detection(self):
        """Run object detection analysis with enhanced visualization"""
        if not self.media_items:
            messagebox.showwarning("警告", "请先添加媒体文件")
            return
            
        model = self.model_var.get()
        if "image" not in SUPPORTED_MODELS[model]["supports"]:
            messagebox.showerror("错误", "当前模型不支持目标检测")
            return
            
        self.toggle_ui_state(tk.DISABLED)
        self.status_label.config(text="正在进行目标检测...")
        self.progress_bar.start()
        
        messages = [{
            "role": "system",
            "content": PROMPT_TEMPLATES["target_detection"]
        }]
        
        # Switch to the visualization tab
        for i in range(self.output_notebook.index("end")):
            if "目标可视化" in self.output_notebook.tab(i, "text"):
                self.output_notebook.select(i)
                break
        
        threading.Thread(target=self._run_detection,
                       args=(messages,)).start()
    
    def process_frame(self, frame, media_item, frame_index):
        """Process a video frame with object detection"""
        try:
            # Prepare model inputs
            model = self.model_var.get()
            
            # Convert frame to base64 for API
            img_byte_arr = BytesIO()
            frame.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            img_data_uri = f"data:image/jpeg;base64,{img_base64}"
            
            # Build message content
            content = [
                {"type": "text", "text": f"分析视频帧 {frame_index}，检测所有可见目标"},
                {"type": "image_url", "image_url": {"url": img_data_uri}}
            ]
            
            # Create API request
            messages = [
                {"role": "system", "content": PROMPT_TEMPLATES["target_detection"]},
                {"role": "user", "content": content}
            ]
            
            # Call API
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature_var.get()
            )
            
            response_text = completion.choices[0].message.content
            
            # Extract annotations from response
            json_data = self.extract_json_from_text(response_text)
            if json_data and "annotations" in json_data:
                # Visualize and add to UI
                annotations = json_data["annotations"]
                
                # Convert PIL image to numpy for visualization
                frame_np = np.array(frame)
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                
                vis_config = {
                    "show_labels": self.show_labels_var.get(),
                    "show_scores": self.show_scores_var.get(),
                    "color_scheme": self.color_scheme_var.get(),
                    "model": model
                }
                
                vis_result = visualize_detection(frame_np, annotations, vis_config)
                
                # Add visualization
                if self.visualization_panel:
                    metadata = {
                        "source": Path(media_item["source"]).name,
                        "frame": frame_index,
                        "objects": len(annotations)
                    }
                    self.visualization_panel.add_visualization(vis_result, metadata)
                
                # Update output text
                self.update_output(f"在帧 {frame_index} 中检测到 {len(annotations)} 个对象\n\n" + response_text)
                
                return annotations
                
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return []
    
    def _run_detection(self, messages):
        """Enhanced detection analysis with real-time visualization"""
        try:
            results = []
            batch_size = 5 if self.batch_mode.get() else 1
            
            for i in range(0, len(self.media_items), batch_size):
                batch = self.media_items[i:i + batch_size]
                content = self.build_content(f"请分析以下{len(batch)}个媒体项，识别并返回所有可见目标")
                
                batch_messages = messages.copy()
                batch_messages.append({
                    "role": "user",
                    "content": content
                })
                
                completion = self.client.chat.completions.create(
                    model=self.model_var.get(),
                    messages=batch_messages,
                    temperature=self.temperature_var.get(),
                    stream=True
                )
                
                buffer = ""
                json_buffer = ""
                json_pattern = re.compile(r'\{.*\}', re.DOTALL)
                
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunk_content = chunk.choices[0].delta.content
                        buffer += chunk_content
                        self.update_output(buffer)
                        
                        # Try to extract JSON content
                        json_match = json_pattern.search(buffer)
                        if json_match and json_match.group() != json_buffer:
                            json_buffer = json_match.group()
                            try:
                                # Process partial JSON
                                json_data = json.loads(json_buffer)
                                if "annotations" in json_data and batch[0]["type"] == "image":
                                    # Process the first image in the batch
                                    self.process_detection_result(batch[0], json_data)
                            except json.JSONDecodeError:
                                # Incomplete JSON, wait for more chunks
                                pass
                
                # Final processing with complete response
                try:
                    json_match = json_pattern.search(buffer)
                    if json_match:
                        final_json = json.loads(json_match.group())
                        if "annotations" in final_json:
                            for media_item in batch:
                                if media_item["type"] == "image":
                                    # Save annotations to media item
                                    media_item["annotations"] = final_json["annotations"]
                                    # Final visualization
                                    self.process_detection_result(media_item, final_json)
                except Exception as e:
                    logger.error(f"JSON processing error: {str(e)}")
                
                results.append(buffer)
                
                # 更新进度
                self.status_label.config(
                    text=f"已处理 {min(i + batch_size, len(self.media_items))}/{len(self.media_items)}"
                )
            
            # 保存分析结果
            self.save_analysis_results(results, "target_detection")
            
        except Exception as e:
            logger.error(f"Detection analysis error: {str(e)}")
            messagebox.showerror("错误", f"目标检测失败: {str(e)}")
        finally:
            self.toggle_ui_state(tk.NORMAL)
            self.status_label.config(text="就绪")
            self.progress_bar.stop()
    
    def process_detection_result(self, media_item, result_json):
        """Process detection results and create visualization"""
        if "annotations" not in result_json or not result_json["annotations"]:
            logger.info(f"没有检测到目标对象，跳过可视化")
            return
        
        try:
            # 获取图像大小调整参数 - 如果界面中没有这些变量，设置默认值
            resize_enabled = getattr(self, "resize_images_var", tk.BooleanVar(value=True)).get()
            max_dimension = getattr(self, "max_image_dimension_var", tk.IntVar(value=1280)).get()
            preserve_aspect_ratio = getattr(self, "preserve_aspect_ratio_var", tk.BooleanVar(value=True)).get()
            save_quality = getattr(self, "save_quality_var", tk.IntVar(value=90)).get()
            
            # 创建增强的可视化配置
            vis_config = {
                "show_labels": self.show_labels_var.get(),
                "show_scores": self.show_scores_var.get(),
                "color_scheme": self.color_scheme_var.get(),
                "model": self.model_var.get(),
                "show_info": True,  # 在图像上显示处理信息
                "show_coordinates": True,  # 显示坐标信息
                # 添加图像大小调整配置
                "resize": {
                    "enabled": resize_enabled,
                    "max_dimension": max_dimension,
                    "preserve_aspect_ratio": preserve_aspect_ratio
                }
            }
            
            # 获取图像源
            if media_item["is_local"]:
                image_source = media_item["source"]
            else:
                # 对于URL，我们可能需要下载
                image_source = media_item.get("original", None)
                if not image_source:
                    logger.error("找不到远程图像源")
                    return
            
            # 记录原始图像大小
            try:
                with Image.open(image_source) as orig_img:
                    orig_size = orig_img.size
                    logger.info(f"原始图像尺寸: {orig_size[0]}x{orig_size[1]}")
            except Exception as e:
                logger.warning(f"无法获取原始图像尺寸: {e}")
            
            # 创建可视化
            logger.info(f"生成检测结果可视化，共有 {len(result_json['annotations'])} 个目标...")
            
            # 准备在窗口打印坐标数据
            coordinate_text = "检测到的目标坐标：\n"
            for i, anno in enumerate(result_json['annotations']):
                if "bbox" in anno:
                    label = anno.get("category", "未知")
                    score = anno.get("score", 1.0)
                    bbox = anno["bbox"]
                    x, y, w, h = [float(v) for v in bbox]
                    coordinate_text += f"{i+1}. {label} ({score:.2f}): 左上({int(x)},{int(y)}), 宽{int(w)}, 高{int(h)}\n"
                    
            # 更新界面中的坐标信息（如果有对应的文本控件）
            if hasattr(self, "coordinates_text"):
                # 清空现有文本
                self.coordinates_text.delete(1.0, tk.END)
                # 插入新坐标文本
                self.coordinates_text.insert(tk.END, coordinate_text)
            
            vis_image = visualize_detection(
                image_source,
                result_json["annotations"],
                vis_config
            )
            
            if vis_image:
                # 记录调整后的图像大小
                resized_size = vis_image.size
                logger.info(f"调整后的图像尺寸: {resized_size[0]}x{resized_size[1]}")
                
                # 添加可视化到面板
                metadata = {
                    "source": os.path.basename(media_item["source"]) if media_item["is_local"] else "URL",
                    "objects": len(result_json["annotations"]),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "dimensions": f"{resized_size[0]}x{resized_size[1]}",  # 添加尺寸信息
                    "resize": "已调整" if resize_enabled and 'orig_size' in locals() and orig_size != resized_size else "原始尺寸"
                }
                
                self.visualization_panel.add_visualization(vis_image, metadata)
                
                # 保存可视化到文件
                save_dir = Path("visualization_results")
                save_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 添加尺寸信息到文件名
                dimension_info = f"{resized_size[0]}x{resized_size[1]}"
                filename = f"detection_{timestamp}_{dimension_info}.png"
                
                # 使用配置的质量设置保存
                vis_image.save(
                    save_dir / filename, 
                    quality=save_quality, 
                    optimize=True  # 优化文件大小
                )
                
                # 计算文件大小并格式化
                saved_file = save_dir / filename
                file_size_bytes = os.path.getsize(saved_file)
                if file_size_bytes > 1024 * 1024:
                    file_size = f"{file_size_bytes / (1024 * 1024):.2f} MB"
                else:
                    file_size = f"{file_size_bytes / 1024:.2f} KB"
                    
                logger.info(f"可视化已保存到 {filename} (大小: {file_size})")
                
                # 如果有状态栏，更新状态信息
                if hasattr(self, "statusbar"):
                    self.statusbar.set_status(f"检测到 {len(result_json['annotations'])} 个目标，已保存可视化 ({file_size})")
                
                # 在单独窗口中显示标注后的图像
                self.show_detection_window(vis_image, coordinate_text, f"检测结果 - {os.path.basename(image_source)}")
        
        except Exception as e:
            logger.error(f"可视化处理错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def show_detection_window(self, image, coordinates_text, title="检测结果"):
        """在单独窗口中显示检测结果及坐标信息
        
        Args:
            image: PIL Image对象
            coordinates_text: 坐标信息文本
            title: 窗口标题
        """
        try:
            # 创建顶层窗口
            detection_window = tk.Toplevel(self.root)
            detection_window.title(title)
            detection_window.geometry("1200x800")
            detection_window.minsize(800, 600)
            
            # 创建分割窗格
            main_pane = ttk.PanedWindow(detection_window, orient=tk.HORIZONTAL)
            main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 左侧图像显示区域
            left_frame = ttk.Frame(main_pane)
            main_pane.add(left_frame, weight=3)
            
            # 右侧坐标信息区域
            right_frame = ttk.Frame(main_pane)
            main_pane.add(right_frame, weight=1)
            
            # 转换PIL图像为Tkinter图像
            img_tk = self.pil_to_tk(image)
            
            # 创建画布并添加滚动条
            canvas_frame = ttk.Frame(left_frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            canvas = tk.Canvas(canvas_frame, bg='white')
            h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
            v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
            
            canvas.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
            
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # 在画布上添加图像
            canvas_image = canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img_ref = img_tk  # 保持引用，防止被垃圾回收
            
            # 设置画布滚动区域
            canvas.config(scrollregion=canvas.bbox(tk.ALL))
            
            # 右侧添加坐标信息文本框
            ttk.Label(right_frame, text="检测目标坐标信息", font=("Arial", 12, "bold")).pack(pady=10)
            
            coord_text = tk.Text(right_frame, wrap=tk.WORD, width=40, height=20)
            coord_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            coord_text.insert(tk.END, coordinates_text)
            
            # 只读模式
            coord_text.config(state=tk.DISABLED)
            
            # 添加右侧滚动条
            coord_scrollbar = ttk.Scrollbar(right_frame, command=coord_text.yview)
            coord_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            coord_text.config(yscrollcommand=coord_scrollbar.set)
            
            # 底部按钮区域
            button_frame = ttk.Frame(detection_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # 添加关闭按钮
            ttk.Button(button_frame, text="关闭窗口", command=detection_window.destroy).pack(side=tk.RIGHT)
            
            # 添加保存按钮
            def save_to_clipboard():
                detection_window.clipboard_clear()
                detection_window.clipboard_append(coordinates_text)
                if hasattr(self, "statusbar"):
                    self.statusbar.set_status("坐标信息已复制到剪贴板")
            
            ttk.Button(button_frame, text="复制坐标到剪贴板", command=save_to_clipboard).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            logger.error(f"创建检测结果窗口错误: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def pil_to_tk(self, pil_image):
        """将PIL图像转换为Tkinter图像"""
        try:
            return ImageTk.PhotoImage(pil_image)
        except Exception as e:
            logger.error(f"图像转换错误: {e}")
            # 创建一个小的错误图像
            error_img = Image.new('RGB', (200, 100), color=(240, 240, 240))
            draw = ImageDraw.Draw(error_img)
            draw.text((10, 40), "图像无法显示", fill=(255, 0, 0))
            return ImageTk.PhotoImage(error_img)
    
    def build_content(self, message):
        """构建包含多媒体内容的消息"""
        if not self.media_items:
            return [{"type": "text", "text": message}]
            
        content = [{"type": "text", "text": message}]
        
        for item in self.media_items:
            if item["type"] == "image":
                try:
                    if item["is_local"]:
                        with open(item["source"], "rb") as f:
                            image_data = f.read()
                            image_base64 = "data:image/jpeg;base64," + base64.b64encode(image_data).decode()
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": image_base64}
                            })
                    else:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": item["source"]}
                        })
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
            
        return content

    def _run_analysis(self, messages, analysis_type):
        """运行分析任务"""
        try:
            results = []
            batch_size = 5 if self.batch_mode.get() else 1
            
            for i in range(0, len(self.media_items), batch_size):
                batch = self.media_items[i:i + batch_size]
                content = self.build_content(f"请分析以下{len(batch)}个媒体项")
                
                messages.append({
                    "role": "user",
                    "content": content
                })
                
                completion = self.client.chat.completions.create(
                    model=self.model_var.get(),
                    messages=messages,
                    temperature=self.temperature_var.get(),
                    stream=True
                )
                
                buffer = ""
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        buffer += chunk.choices[0].delta.content
                        self.update_output(buffer)
                
                results.append(buffer)
                
                # 更新进度
                self.status_label.config(
                    text=f"已处理 {min(i + batch_size, len(self.media_items))}/{len(self.media_items)}"
                )
            
            # 保存分析结果
            self.save_analysis_results(results, analysis_type)
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            messagebox.showerror("错误", f"分析失败: {str(e)}")
        finally:
            self.toggle_ui_state(tk.NORMAL)
            self.status_label.config(text="就绪")
            self.progress_bar.stop()

    def update_output(self, text):
        """更新输出文本框"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def save_analysis_results(self, results, analysis_type):
        """
        保存分析结果到文件系统
        
        Args:
            results: 分析结果列表
            analysis_type: 分析类型（例如："target_detection"）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"analysis_results/{analysis_type}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录有关此次分析的元数据
        metadata = {
            "type": analysis_type,
            "timestamp": timestamp,
            "results": results,
            "model": self.model_var.get(),
            "settings": {
                "temperature": self.temperature_var.get(),
                "batch_mode": self.batch_mode.get()
            }
        }
        
        # 保存JSON格式结果
        json_path = save_dir / f"{timestamp}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存到 {json_path}")
        except Exception as e:
            logger.error(f"保存JSON结果失败: {str(e)}")
            return False
        
        # 只有目标检测类型的结果需要生成可视化
        if analysis_type != "target_detection" or not results:
            return True
        
        # 创建可视化保存目录
        vis_dir = save_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 获取用户界面的配置参数
        vis_config = {
            "show_labels": getattr(self, "show_labels_var", tk.BooleanVar(value=True)).get(),
            "show_scores": getattr(self, "show_scores_var", tk.BooleanVar(value=True)).get(),
            "color_scheme": getattr(self, "color_scheme_var", tk.StringVar(value="default")).get(),
            "model": self.model_var.get()
        }
        
        # 获取图像大小调整参数
        resize_enabled = getattr(self, "resize_images_var", tk.BooleanVar(value=True)).get()
        max_dimension = getattr(self, "max_image_dimension_var", tk.IntVar(value=1280)).get()
        preserve_aspect_ratio = getattr(self, "preserve_aspect_ratio_var", tk.BooleanVar(value=True)).get()

        # 保存可视化图像
        success_count = 0
        error_count = 0
        for i, (result, media_item) in enumerate(zip(results, self.media_items)):
            # 只处理图像类型
            if media_item["type"] != "image":
                continue
                
            try:
                # 从结果中提取JSON数据 - 处理多种可能的格式
                annotations = None
                if isinstance(result, dict):
                    # 结果已经是字典
                    annotations = result.get("annotations", None)
                elif isinstance(result, str):
                    # 尝试从字符串中提取JSON
                    try:
                        # 首先尝试解析整个字符串
                        json_result = json.loads(result)
                        if isinstance(json_result, dict):
                            annotations = json_result.get("annotations", None)
                    except json.JSONDecodeError:
                        # 如果失败，尝试使用正则表达式提取
                        json_pattern = re.compile(r'\{.*\}', re.DOTALL)
                        json_match = json_pattern.search(result)
                        if json_match:
                            try:
                                json_data = json.loads(json_match.group())
                                annotations = json_data.get("annotations", None)
                            except json.JSONDecodeError:
                                pass
                
                # 如果找到了注释数据，生成可视化
                if annotations:
                    # 加载图像（根据source类型适配不同的加载方式）
                    source = media_item["source"]
                    
                    # 生成可视化图像
                    img = visualize_detection(source, annotations, vis_config)
                    
                    if img:
                        # 生成保存路径
                        file_name = f"{timestamp}_{i}.png"
                        save_path = vis_dir / file_name
                        
                        # 保存图像
                        img.save(save_path)
                        success_count += 1
                        logger.debug(f"已保存可视化图像: {file_name}")
            
            except Exception as e:
                error_count += 1
                logger.error(f"生成第 {i+1} 个结果的可视化时出错: {str(e)}")
        
        # 记录保存结果的统计信息
        total = len(results)
        logger.info(f"目标检测可视化结果: 成功 {success_count}/{total}, 失败 {error_count}/{total}")
        
        return success_count > 0

    def send_message(self):
        """Handle sending chat messages"""
        message = self.chat_entry.get().strip()
        if not message:
            return
            
        self.chat_entry.delete(0, tk.END)
        self.toggle_ui_state(tk.DISABLED)
        self.status_label.config(text="正在处理...")
        self.progress_bar.start()
        
        # Prepare messages for API call
        messages = []
        if self.prompt_text.get(1.0, tk.END).strip():
            messages.append({
                "role": "system",
                "content": self.prompt_text.get(1.0, tk.END).strip()
            })
            
        # Add context from conversation history
        for entry in self.conversation[-4:]:  # Include last 4 exchanges for context
            messages.append(entry)
        # Add current message
        content = self.build_content(message)
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Start API call in a separate thread
        threading.Thread(target=self.call_api,
                       args=(messages, self.model_var.get())).start()

    def call_api(self, messages, model_name, is_detection=False):
        """API调用处理"""
        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.temperature_var.get(),
                stream=True
            )

            buffer = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    buffer += chunk.choices[0].delta.content
                    self.update_output(buffer)
                    
                    # 实时处理目标检测结果
                    if is_detection:
                        try:
                            json_pattern = re.compile(r'\{.*\}', re.DOTALL)
                            json_match = json_pattern.search(buffer)
                            if json_match:
                                json_data = json.loads(json_match.group())
                                if "annotations" in json_data:
                                    for media in self.media_items:
                                        if media["type"] == "image":
                                            media["annotations"] = json_data["annotations"]
                                            self.process_detection_result(media, json_data)
                        except json.JSONDecodeError:
                            pass  # 等待更多数据

            # 保存到历史记录
            self.conversation.append({"role": "user", "content": messages[-1]["content"]})
            self.conversation.append({"role": "assistant", "content": buffer})
            self.update_history()
            
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            self.update_output(f"\nAPI错误: {e.message}")
        except Exception as e:
            logger.error(f"System error: {str(e)}")
            self.update_output(f"\n系统错误: {str(e)}")
        finally:
            self.toggle_ui_state(tk.NORMAL)
            self.status_label.config(text="就绪")
            self.progress_bar.stop()

    def update_history(self):
        """更新历史记录显示"""
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        for entry in self.conversation:
            role = "用户" if entry["role"] == "user" else "助手"
            content = entry["content"]
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            
            self.history_text.insert(tk.END, f"\n{role}:\n{content}\n")
            self.history_text.insert(tk.END, "-" * 50 + "\n")
        
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)

    def export_results(self):
        """导出分析结果"""
        if not self.conversation:
            messagebox.showwarning("警告", "没有可导出的结果")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP archives", "*.zip"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            import zipfile
            from io import StringIO
            
            with zipfile.ZipFile(file_path, "w") as zf:
                # 导出对话记录
                chat_buffer = StringIO()
                for entry in self.conversation:
                    role = "用户" if entry["role"] == "user" else "助手"
                    content = entry["content"]
                    if isinstance(content, list):
                        content = "\n".join(str(item) for item in content)
                    chat_buffer.write(f"\n{role}:\n{content}\n{'='*50}\n")
                
                zf.writestr("conversation.txt", chat_buffer.getvalue())
                
                # 导出媒体分析结果
                for i, media in enumerate(self.media_items):
                    if "annotations" in media:
                        zf.writestr(
                            f"annotations/item_{i}.json",
                            json.dumps(media["annotations"], ensure_ascii=False, indent=2)
                        )
                        
                    if media["type"] == "image" and "annotations" in media:
                        vis_config = {
                            "show_labels": self.show_labels_var.get(),
                            "show_scores": self.show_scores_var.get(),
                            "color_scheme": self.color_scheme_var.get(),
                            "model": self.model_var.get()
                        }
                        img = visualize_detection(
                            media["source"],
                            media["annotations"],
                            vis_config
                        )
                        if img:
                            img_buffer = BytesIO()
                            img.save(img_buffer, format="PNG")
                            zf.writestr(f"visualizations/item_{i}.png", img_buffer.getvalue())
                
                # Export video tracking results if available
                if self.video_processor and self.video_processor.tracker.track_history:
                    tracking_data = {
                        "tracks": {},
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "model": self.model_var.get(),
                            "tracker_algorithm": self.tracking_algo_var.get()
                        }
                    }
                    
                    # Convert track history to serializable format
                    for track_id, history in self.video_processor.tracker.track_history.items():
                        tracking_data["tracks"][str(track_id)] = [list(bbox) for bbox in history]
                        
                    zf.writestr(
                        "tracking_results.json",
                        json.dumps(tracking_data, ensure_ascii=False, indent=2)
                    )
                
            messagebox.showinfo("成功", f"结果已导出到: {file_path}")
            
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            messagebox.showerror("导出错误", str(e))

    def extract_json_from_text(self, text):
        """Extract JSON content from text response"""
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)
        json_match = json_pattern.search(text)
        
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None

# Main function to run the application
def main():
    try:
        # Check for API key in environment
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            # Will prompt for API key when app starts
            pass
        
        # Configure application style
        root = tk.Tk()
        root.title("智能多模态分析系统 v4.0 - 视频跟踪版")
        
        # Set app icon if available
        try:
            icon_path = Path("assets/app_icon.ico")
            if icon_path.exists():
                root.iconbitmap(str(icon_path))
        except Exception:
            pass  # No icon, continue anyway
        
        # Set default font
        default_font = ('Microsoft YaHei UI', 11)
        root.option_add("*Font", default_font)
        
        # Add DPI awareness on Windows
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
        
        # Create application
        app = MultimodalApp(root)
        
        # Register keyboard shortcuts
        root.bind("<Control-n>", lambda e: app.new_session())
        root.bind("<Control-s>", lambda e: app.save_session())
        root.bind("<Control-e>", lambda e: app.export_results())
        root.bind("<Control-q>", lambda e: root.quit())
        root.bind("<space>", lambda e: app.pause_video() if app.is_playing_video else app.play_video())
        
        # Center the window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("致命错误", f"程序遇到无法恢复的错误:\n{str(e)}")

if __name__ == "__main__":
    main()