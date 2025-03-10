import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

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
                        # Convert PhotoImage back to PIL and save
                        image = self.video_label._image
                        image.save(file_path)
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