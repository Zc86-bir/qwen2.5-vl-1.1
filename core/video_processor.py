import cv2
import numpy as np
import logging
import threading
import time
import json
import base64
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime

from config import TRACKING_ALGORITHMS
import tracking_utils

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process videos for AI analysis and tracking"""
    
    def __init__(self, client=None, model=None, config=None):
        self.client = client
        self.model_name = model or "qwen-vl-max"
        self.config = config or {}
        
        # Get tracker algorithm name
        self.tracker_algorithm = self.config.get("tracking_algorithm", "Basic")
        
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
        self.trackers = {}  # {track_id: tracker}
        self.next_track_id = 0
        
        # Thread management
        self.processing_thread = None
        self.detection_interval = self.config.get("detection_interval", 20)  # Run detection every 20 frames
        self.result_queue = Queue()
        
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
        self.trackers = {}
        
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
    
    def _create_tracker(self):
        """创建跟踪器实例"""
        from tracking_utils import create_tracker
        tracker = create_tracker(self.tracker_algorithm)
        if tracker is None:
            logger.error(f"无法创建跟踪器: {self.tracker_algorithm}")
            # 尝试基础跟踪器
            tracker = create_tracker("Basic")
            
        return tracker
            
    def _add_tracker(self, frame, bbox, label=None):
        """Add a new tracker for an object"""
        tracker = self._create_tracker()
        if tracker is None:
            logger.error("Failed to create tracker")
            return -1
            
        # Convert from [x1, y1, x2, y2] format to [x, y, w, h] if needed
        if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            # [x1, y1, x2, y2] format
            x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        else:
            x, y, w, h = bbox
            
        try:
            success = tracker.init(frame, (x, y, w, h))
            
            if success:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.trackers[track_id] = tracker
                self.tracks[track_id] = (x, y, w, h)
                
                # Initialize track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append((x, y, w, h))
                
                logger.debug(f"Added tracker {track_id} for bbox {(x, y, w, h)}")
                return track_id
            else:
                logger.warning(f"Failed to initialize tracker for bbox {bbox}")
                return -1
                
        except Exception as e:
            logger.error(f"Error initializing tracker: {str(e)}")
            return -1
    
    def start_processing(self, callback=None):
        """Start video processing in a separate thread"""
        if self.is_processing:
            logger.warning("Video processing already running")
            return False
            
        if not self.current_video or not self.current_video.isOpened():
            logger.error("No video loaded")
            return False
        
        # Reset tracking state
        self.tracks = {}
        self.track_history = {}
        self.trackers = {}
        self.next_track_id = 0
        
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
            current_tracks = {}
            tracks_to_remove = []
            
            for track_id, tracker in self.trackers.items():
                try:
                    success, bbox = tracker.update(frame)
                    
                    if success:
                        # Convert bbox to integers
                        x, y, w, h = [int(v) for v in bbox]
                        current_tracks[track_id] = (x, y, w, h)
                        
                        # Update track history
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append((x, y, w, h))
                        
                        # Limit history length
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id].pop(0)
                    else:
                        tracks_to_remove.append(track_id)
                        
                except Exception as e:
                    logger.error(f"Error updating tracker {track_id}: {str(e)}")
                    tracks_to_remove.append(track_id)
            
            # Remove failed trackers
            for track_id in tracks_to_remove:
                if track_id in self.trackers:
                    del self.trackers[track_id]
                if track_id in self.tracks:
                    del self.tracks[track_id]
            
            # Update current tracks
            self.tracks = current_tracks
            
            # Run detection with the AI model if needed
            if run_detection:
                detection_annotations = self._detect_objects(frame)
                if detection_annotations:
                    last_detection_frame = frame_index
                    self.detections = detection_annotations
                    
                    # Add new trackers for detections
                    for detection in detection_annotations:
                        bbox = detection.get("bbox_2d")
                        label = detection.get("label", "object")
                        
                        if not bbox or len(bbox) != 4:
                            continue
                            
                        # Initialize tracker
                        self._add_tracker(frame, bbox, label)
            
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
                "annotations": self._get_annotations()
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
    
    def _get_annotations(self):
        """Get current tracking results as annotations"""
        annotations = []
        
        for track_id, bbox in self.tracks.items():
            x, y, w, h = bbox
            
            annotation = {
                "bbox_2d": [x, y, x+w, y+h],  # Convert to [x1, y1, x2, y2] format
                "label": f"Object {track_id}",
                "track_id": track_id
            }
            
            annotations.append(annotation)
            
        return annotations
    
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
        except Empty:
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
        for track_id, bbox in self.tracks.items():
            # Get color for this track
            color_idx = track_id % len(colors)
            color = colors[color_idx]
            
            # Extract bbox coordinates
            x, y, w, h = bbox
            
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
                    "tracker_algorithm": self.tracker_algorithm or "unknown"
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
            
    def get_available_trackers(self):
        """Return list of available trackers"""
        return list(TRACKING_ALGORITHMS.keys())