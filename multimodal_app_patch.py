"""
Patch for multimodal_app.py to properly handle tracker integration
Apply this in your main code or integrate these changes into your MultimodalApp class
"""

import tracking_utils

def patch_video_processor(video_processor):
    """Patch an existing video processor to use the tracking_utils properly"""
    
    # Override the _create_tracker method to use our tracking_utils
    def patched_create_tracker(self):
        """Create tracker using tracking_utils factory"""
        return tracking_utils.create_tracker(self.tracker_algorithm)
    
    # Replace the method
    video_processor._create_tracker = patched_create_tracker.__get__(video_processor)

def get_available_trackers():
    """Get list of available tracker algorithms from tracking_utils"""
    return tracking_utils.get_available_algorithms()

# Add to your main code
def apply_tracking_patches():
    """Apply all necessary tracking patches"""
    # This should be called early in your application initialization
    from tracking_utils import get_available_algorithms
    
    # Print available algorithms for debugging
    available = get_available_algorithms()
    print(f"Available tracking algorithms from tracking_utils: {available}")
    
    # Make sure the first tracker is always 'Basic' (guaranteed to work)
    if 'Basic' not in available:
        print("Warning: Basic tracker should always be available")