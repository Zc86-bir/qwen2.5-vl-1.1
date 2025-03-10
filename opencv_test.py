import cv2
import numpy as np

print(f"OpenCV 版本: {cv2.__version__}")
print(f"是否包含 legacy 模块: {hasattr(cv2, 'legacy')}")

# 创建测试图像
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
test_bbox = (10, 10, 30, 30)  # x, y, w, h

# 检测跟踪器
trackers = []

# 测试方法1: 从 cv2 直接创建
for name in ["CSRT", "KCF", "MIL", "MOSSE", "MedianFlow", "BOOSTING", "TLD"]:
    try:
        if hasattr(cv2, f"Tracker{name}_create"):
            tracker = getattr(cv2, f"Tracker{name}_create")()
            if tracker.init(test_img, test_bbox):
                trackers.append(f"{name}")
                print(f"✓ 成功创建 {name} 跟踪器 (cv2 直接)")
    except Exception as e:
        print(f"✗ cv2.Tracker{name}_create 失败: {str(e)}")

# 测试方法2: 从 legacy 模块创建
if hasattr(cv2, 'legacy'):
    for name in ["CSRT", "KCF", "MIL", "MOSSE", "MedianFlow", "BOOSTING", "TLD"]:
        try:
            if hasattr(cv2.legacy, f"Tracker{name}_create"):
                tracker = getattr(cv2.legacy, f"Tracker{name}_create")()
                if tracker.init(test_img, test_bbox):
                    if name not in trackers:
                        trackers.append(f"{name}")
                        print(f"✓ 成功创建 {name} 跟踪器 (legacy)")
        except Exception as e:
            print(f"✗ legacy.Tracker{name}_create 失败: {str(e)}")

print(f"\n可用跟踪器: {', '.join(trackers) if trackers else '无'}")