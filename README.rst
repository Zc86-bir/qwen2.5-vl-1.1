# 智能多模态分析系统 (Intelligent Multimodal Analysis System)

一个功能强大的多模态分析工具，支持图像分析、视频目标跟踪和智能对话功能。

## 功能特点

- **图像分析**: 目标检测、场景理解、文本提取
- **视频分析**: 视频目标跟踪、轨迹可视化
- **智能对话**: 基于大语言模型的多模态对话
- **批处理能力**: 支持批量导入和分析媒体文件
- **可视化**: 高级可视化和标注功能

## 系统要求

- Python 3.8 或更高版本
- 以下第三方库:
  - openai
  - opencv-python
  - pillow
  - numpy
  - tenacity
  - requests

## 快速开始

1. 克隆此仓库或下载源代码
2. 设置 API 密钥（DashScope API Key）
   ```bash
   # Linux/Mac
   export DASHSCOPE_API_KEY=your_api_key_here
   
   # Windows
   set DASHSCOPE_API_KEY=your_api_key_here
   ```

3. 运行启动脚本
   ```bash
   python run_app.py
   ```
   启动脚本会自动检查并安装缺少的依赖库。

## 使用方法

### 图像分析

1. 点击"上传文件"按钮导入图片
2. 选择合适的模型和分析类型（目标定位/场景分析/文本提取）
3. 点击相应的分析按钮
4. 查看"分析结果"选项卡中的结果
5. 在"目标可视化"选项卡查看带标注的图像

### 视频分析

1. 点击"上传视频"按钮导入视频
2. 使用视频控制按钮播放视频
3. 点击"启动跟踪"开始目标跟踪
4. 在"视频分析"选项卡查看实时跟踪效果
5. 使用"导出轨迹"保存