import cv2
import numpy as np
import logging
from pathlib import Path
import json
import time
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class ObjectTracker:
    """Class to handle object tracking in videos with multiple algorithm support"""
    
    # Define available tracking algorithms with their creation methods and properties
    AVAILABLE_TRACKERS = {
        # Classic trackers
        "CSRT": {
            "create": lambda: cv2.legacy.TrackerCSRT_create() if hasattr(cv2, "legacy") else cv2.TrackerCSRT_create() if hasattr(cv2, "TrackerCSRT_create") else None,
            "description": "Discriminative Correlation Filter with Channel and Spatial Reliability. Accurate but slower.",
            "properties": {
                "accuracy": "High",
                "speed": "Low",
                "occlusion_handling": "Good",
                "use_case": "When accuracy is more important than speed"
            }
        },
        "KCF": {
            "create": lambda: cv2.legacy.TrackerKCF_create() if hasattr(cv2, "legacy") else cv2.TrackerKCF_create() if hasattr(cv2, "TrackerKCF_create") else None,
            "description": "Kernelized Correlation Filters. Good balance between accuracy and speed.",
            "properties": {
                "accuracy": "Medium",
                "speed": "Medium",
                "occlusion_handling": "Fair",
                "use_case": "Balanced performance"
            }
        },
        "MOSSE": {
            "create": lambda: cv2.legacy.TrackerMOSSE_create() if hasattr(cv2, "legacy") else cv2.TrackerMOSSE_create() if hasattr(cv2, "TrackerMOSSE_create") else None,
            "description": "Minimum Output Sum of Squared Error. Very fast but less accurate.",
            "properties": {
                "accuracy": "Low",
                "speed": "Very High",
                "occlusion_handling": "Poor",
                "use_case": "When speed is critical"
            }
        },
        # More advanced trackers
        "MedianFlow": {
            "create": lambda: cv2.legacy.TrackerMedianFlow_create() if hasattr(cv2, "legacy") else cv2.TrackerMedianFlow_create() if hasattr(cv2, "TrackerMedianFlow_create") else None,
            "description": "Tracks the object by minimizing the forward-backward error. Good for predictable motion.",
            "properties": {
                "accuracy": "Medium",
                "speed": "High",
                "occlusion_handling": "Poor",
                "use_case": "Smooth, predictable motion"
            }
        },
        "MIL": {
            "create": lambda: cv2.legacy.TrackerMIL_create() if hasattr(cv2, "legacy") else cv2.TrackerMIL_create() if hasattr(cv2, "TrackerMIL_create") else None,
            "description": "Multiple Instance Learning. Handles partial occlusion well.",
            "properties": {
                "accuracy": "Medium",
                "speed": "Medium",
                "occlusion_handling": "Good",
                "use_case": "Scenes with partial occlusions"
            }
        },
        # Custom implementation: Simple IoU tracker
        "IoU": {
            "create": lambda: IoUTracker(),
            "description": "Simple IoU (Intersection over Union) based tracking. Works with detection output only.",
            "properties": {
                "accuracy": "Depends on detector",
                "speed": "High",
                "occlusion_handling": "Poor",
                "use_case": "When continuous detection is available"
            }
        }
    }
    
    def __init__(self, algorithm="CSRT"):
        """Initialize tracker with specified algorithm"""
        self.algorithm = algorithm
        
        # Fall back if algorithm not available
        if algorithm not in self.AVAILABLE_TRACKERS or self.AVAILABLE_TRACKERS[algorithm]["create"]() is None:
            logger.warning(f"Tracker {algorithm} not available. Falling back to CSRT.")
            available = [name for name, info in self.AVAILABLE_TRACKERS.items() 
                         if info["create"]() is not None]
            if available:
                self.algorithm = available[0]
            else:
                raise RuntimeError("No tracking algorithms available in this OpenCV build")
            
        logger.info(f"Using tracking algorithm: {self.algorithm}")
        
        self.trackers = {}
        self.next_id = 0
        self.track_history = defaultdict(list)  # Store tracking history for visualization
        self.max_history_len = 30  # Maximum number of history points to keep per track
        
        # For IoU tracker
        self.prev_detections = {}
        self.iou_threshold = 0.5
        
    def init_tracker(self, frame, bbox, label=None):
        """Initialize a new tracker for an object"""
        if self.algorithm not in self.AVAILABLE_TRACKERS:
            logger.error(f"Tracking algorithm {self.algorithm} not available")
            return -1
        
        try:
            if self.algorithm == "IoU":
                # IoU tracker doesn't need initialization
                track_id = self.next_id
                self.next_id += 1
                
                # Store as detection
                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    # [x1, y1, x2, y2] format
                    self.prev_detections[track_id] = {
                        "bbox": bbox,
                        "label": label or "object",
                        "last_seen": 0
                    }
                else:
                    # [x, y, w, h] format
                    x, y, w, h = bbox
                    self.prev_detections[track_id] = {
                        "bbox": [x, y, x+w, y+h],
                        "label": label or "object",
                        "last_seen": 0
                    }
                
                # Initialize track history
                self.track_history[track_id].append(bbox if len(bbox) == 4 else (bbox[0], bbox[1], bbox[2], bbox[3]))
                return track_id
                
            else:
                # Create tracker instance
                tracker_create = self.AVAILABLE_TRACKERS[self.algorithm]["create"]
                if tracker_create is None:
                    logger.error(f"Failed to create {self.algorithm} tracker")
                    return -1
                
                tracker = tracker_create()
                
                # Convert from [x1, y1, x2, y2] to [x, y, w, h] format
                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:  
                    # [x1, y1, x2, y2] format
                    x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
                elif len(bbox) == 4:  
                    # [x, y, w, h] format
                    x, y, w, h = bbox
                else:
                    logger.error(f"Invalid bbox format: {bbox}")
                    return -1
                
                success = tracker.init(frame, (x, y, w, h))
                if not success:
                    logger.error("Failed to initialize tracker")
                    return -1
                
                # Assign a unique ID and store the tracker
                track_id = self.next_id
                self.next_id += 1
                
                self.trackers[track_id] = {
                    "tracker": tracker,
                    "bbox": (x, y, w, h),
                    "label": label or "object",
                    "frame_count": 0,
                    "consecutive_failures": 0
                }
                
                # Initialize track history
                self.track_history[track_id].append((x, y, w, h))
                
                logger.info(f"Initialized tracker {track_id} for {label or 'object'}")
                return track_id
                
        except Exception as e:
            logger.error(f"Error initializing tracker: {str(e)}")
            return -1
    
    def update_trackers(self, frame):
        """Update all trackers with a new frame"""
        results = {}
        trackers_to_remove = []
        
        if self.algorithm == "IoU":
            # IoU tracker is updated with detections, not frame by frame
            # Just return current detections
            for track_id, detection in self.prev_detections.items():
                bbox = detection["bbox"]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    results[track_id] = (x1, y1, x2-x1, y2-y1)  # Convert to x,y,w,h
                
                # Increment "last seen" counter
                detection["last_seen"] += 1
                
                # Remove tracks not updated for too long
                if detection["last_seen"] > 30:  # If not seen for 30 frames
                    trackers_to_remove.append(track_id)
                    
            # Remove stale tracks
            for track_id in trackers_to_remove:
                del self.prev_detections[track_id]
                
            return results
            
        # Standard OpenCV tracker update
        for track_id, track_info in self.trackers.items():
            tracker = track_info["tracker"]
            try:
                success, bbox = tracker.update(frame)
                
                if success:
                    # Update the bbox in our tracker info
                    x, y, w, h = [int(v) for v in bbox]
                    self.trackers[track_id]["bbox"] = (x, y, w, h)
                    self.trackers[track_id]["consecutive_failures"] = 0
                    results[track_id] = (x, y, w, h)
                    
                    # Update track history
                    self.track_history[track_id].append((x, y, w, h))
                    if len(self.track_history[track_id]) > self.max_history_len:
                        self.track_history[track_id].pop(0)
                else:
                    # Track failed
                    self.trackers[track_id]["consecutive_failures"] += 1
                    
                    # If failed too many times, mark for removal
                    if self.trackers[track_id]["consecutive_failures"] >= 10:
                        trackers_to_remove.append(track_id)
                    
                # Increment frame count
                self.trackers[track_id]["frame_count"] += 1
                
            except Exception as e:
                logger.error(f"Error updating tracker {track_id}: {str(e)}")
                trackers_to_remove.append(track_id)
        
        # Remove failed trackers
        for track_id in trackers_to_remove:
            logger.info(f"Removing tracker {track_id} after consecutive failures")
            if track_id in self.trackers:
                del self.trackers[track_id]
            
        return results
        
    def add_detections(self, frame, detections):
        """Add new detections as trackers or update IoU tracker"""
        new_track_ids = []
        
        if self.algorithm == "IoU":
            # For IoU tracking, match with previous detections
            current_detections = {}
            
            # Convert detections to the right format
            for det in detections:
                bbox = det.get("bbox_2d")
                label = det.get("label", "object")
                
                if not bbox or len(bbox) != 4:
                    continue
                    
                current_detections[len(current_detections)] = {
                    "bbox": bbox,
                    "label": label
                }
            
            # Associate detections with tracks
            matched_tracks, unmatched_detections = self._associate_detections_to_tracks(current_detections)
            
            # Update matched tracks
            for track_id, det_id in matched_tracks:
                self.prev_detections[track_id]["bbox"] = current_detections[det_id]["bbox"]
                self.prev_detections[track_id]["last_seen"] = 0  # Reset counter
                
                # Update track history
                bbox = current_detections[det_id]["bbox"]
                x1, y1, x2, y2 = bbox
                self.track_history[track_id].append((x1, y1, x2-x1, y2-y1))
                if len(self.track_history[track_id]) > self.max_history_len:
                    self.track_history[track_id].pop(0)
                    
                new_track_ids.append(track_id)
            
            # Create new tracks for unmatched detections
            for det_id in unmatched_detections:
                bbox = current_detections[det_id]["bbox"]
                label = current_detections[det_id]["label"]
                
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                self.prev_detections[track_id] = {
                    "bbox": bbox,
                    "label": label,
                    "last_seen": 0
                }
                
                # Initialize track history
                x1, y1, x2, y2 = bbox
                self.track_history[track_id].append((x1, y1, x2-x1, y2-y1))
                
                new_track_ids.append(track_id)
                
            return new_track_ids
            
        else:
            # Standard tracking: initialize new trackers for each detection
            for detection in detections:
                bbox = detection.get("bbox_2d")
                label = detection.get("label", "object")
                
                if not bbox or len(bbox) != 4:
                    continue
                    
                # Initialize a new tracker for this detection
                track_id = self.init_tracker(frame, bbox, label)
                if track_id >= 0:
                    new_track_ids.append(track_id)
            
            return new_track_ids
    
    def _associate_detections_to_tracks(self, detections):
        """Associate detections with existing tracks using IoU"""
        if not self.prev_detections or not detections:
            return [], list(detections.keys())
            
        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.prev_detections), len(detections)))
        
        for i, (track_id, track_info) in enumerate(self.prev_detections.items()):
            track_bbox = track_info["bbox"]
            
            for j, (det_id, det_info) in enumerate(detections.items()):
                det_bbox = det_info["bbox"]
                
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        
        # Apply Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        except ImportError:
            # Fallback to greedy assignment if scipy not available
            track_indices, det_indices = self._greedy_assignment(iou_matrix)
            
        # Create matches, filter by threshold
        matches = []
        unmatched_detections = list(detections.keys())
        
        track_ids = list(self.prev_detections.keys())
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                track_id = track_ids[track_idx]
                det_id = list(detections.keys())[det_idx]
                matches.append((track_id, det_id))
                if det_id in unmatched_detections:
                    unmatched_detections.remove(det_id)
        
        return matches, unmatched_detections
    
    def _greedy_assignment(self, iou_matrix):
        """Simple greedy assignment algorithm as fallback"""
        track_indices = []
        det_indices = []
        
        # Create flattened indices sorted by IoU
        flat_indices = np.argsort(-iou_matrix.flatten())
        rows, cols = iou_matrix.shape
        row_indices = flat_indices // cols
        col_indices = flat_indices % cols
        
        assigned_rows = set()
        assigned_cols = set()
        
        # Assign greedily
        for row, col in zip(row_indices, col_indices):
            if row not in assigned_rows and col not in assigned_cols:
                track_indices.append(row)
                det_indices.append(col)
                assigned_rows.add(row)
                assigned_cols.add(col)
                
                # Stop when we've assigned all possible matches
                if len(track_indices) == min(rows, cols):
                    break
        
        return np.array(track_indices), np.array(det_indices)
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate intersection over union between two bboxes"""
        # Ensure bboxes are in [x1, y1, x2, y2] format
        if len(bbox1) == 4 and len(bbox2) == 4:
            # Calculate intersection
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            # Check for non-overlapping boxes
            if x1 >= x2 or y1 >= y2:
                return 0.0
                
            # Calculate areas
            intersection_area = (x2 - x1) * (y2 - y1)
            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            
            # Calculate IoU
            iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
            return iou
        else:
            return 0.0
    
    def get_annotations(self):
        """Get current tracking results as annotations"""
        annotations = []
        
        if self.algorithm == "IoU":
            # For IoU tracker, convert detections to annotations
            for track_id, detection in self.prev_detections.items():
                bbox = detection["bbox"]
                
                annotation = {
                    "bbox_2d": bbox,  # Already in [x1, y1, x2, y2] format
                    "label": detection["label"],
                    "track_id": track_id
                }
                
                annotations.append(annotation)
        else:
            # For standard trackers
            for track_id, track_info in self.trackers.items():
                x, y, w, h = track_info["bbox"]
                
                annotation = {
                    "bbox_2d": [x, y, x+w, y+h],  # Convert to [x1, y1, x2, y2] format
                    "label": track_info["label"],
                    "track_id": track_id
                }
                
                annotations.append(annotation)
            
        return annotations
    
    def reset(self):
        """Reset all trackers"""
        self.trackers = {}
        self.prev_detections = {}
        self.track_history = defaultdict(list)
        logger.info("All trackers reset")
        
    @classmethod
    def get_available_algorithms(cls):
        """Return list of available tracking algorithms"""
        available = []
        for name, info in cls.AVAILABLE_TRACKERS.items():
            try:
                tracker = info["create"]()
                if tracker is not None or name == "IoU":  # IoU tracker is always available
                    available.append({
                        "name": name,
                        "description": info["description"],
                        "properties": info["properties"]
                    })
            except:
                pass
        return available


class IoUTracker:
    """Simple placeholder class for IoU tracker"""
    def __init__(self):
        pass
        
    def init(self, frame, bbox):
        return True
        
    def update(self, frame):
        # IoU tracker doesn't work this way, but we return success and empty bbox
        # to maintain the interface
        return True, [0, 0, 0, 0]