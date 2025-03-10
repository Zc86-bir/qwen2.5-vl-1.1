import os
import base64
import logging
import requests
import io
import json
from PIL import Image
from typing import List, Dict, Any, Union
import time

logger = logging.getLogger(__name__)

class ModelAPIClient:
    """大模型API客户端"""
    
    def __init__(self, api_key=None, model_name=None):
        """初始化API客户端
        
        Parameters:
            api_key: API密钥（如果为None则尝试从环境变量获取）
            model_name: 模型名称
        """
        # 获取API密钥
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            logger.warning("未找到API密钥，请设置DASHSCOPE_API_KEY环境变量或在初始化时提供")
        
        # 修正模型名称 - DashScope API的正确模型标识符
        if model_name and model_name == "qwen2.5-vl-72b-instruct":
            # 修正为DashScope API接受的实际模型名称
            self.model_name = "qwen-vl-max"  # 这里使用最新的通用VL大模型
        else:
            self.model_name = model_name or "qwen-vl-max"  # 默认使用qwen-vl-max
        
        logger.info(f"使用模型: {self.model_name}")
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

        
    def _encode_image(self, image: Union[str, Image.Image]) -> str:
        """将图像编码为base64字符串
        
        Parameters:
            image: PIL.Image对象或图像文件路径
                
        Returns:
            base64编码的图像字符串
        """
        try:
            if isinstance(image, str):
                # 如果是文件路径
                with open(image, "rb") as f:
                    img_bytes = f.read()
            else:
                # 如果是PIL.Image对象
                buffer = io.BytesIO()
                # 确保使用JPEG格式，质量80%以减小大小
                image.save(buffer, format="JPEG", quality=80, optimize=True)
                img_bytes = buffer.getvalue()
            
            # 编码为base64
            base64_data = base64.b64encode(img_bytes).decode("utf-8")
            
            # 添加DashScope需要的data URI前缀
            return f"data:image/jpeg;base64,{base64_data}"
        except Exception as e:
            logger.error(f"编码图像出错: {e}")
            raise
            
    def analyze_video_frames(self, prompt: str, frames: List[Dict[str, Any]]) -> str:
        """分析视频帧内容
        
        Parameters:
            prompt: 提示词
            frames: 帧数据列表，每个元素包含image和position
            
        Returns:
            模型生成的分析结果
        """
        if not self.api_key:
            return "错误: 未配置API密钥"
            
        try:
            # 构建多模态消息
            messages = []
            
            # 系统角色消息
            messages.append({
                "role": "system", 
                "content": [
                    {
                        "text": """你是一个专业的视频分析助手。请基于提供的视频关键帧序列进行全面分析：
    1. 视频内容概述：简要描述视频的主要内容、场景和人物
    2. 时间线分析：按时间顺序分析关键事件和场景变化
    3. 技术评估：评估视频的质量、构图、光线等技术元素
    4. 见解总结：提供对视频内容的专业见解和建议

    请注意每个图像都有附带的时间戳信息。"""
                    }
                ]
            })
            
            # 用户消息内容 - 严格按照每个元素只有一个模态
            user_content = []
            
            # 添加提示文本作为第一个内容
            user_content.append({"text": prompt})
            
            # 添加每个帧的图像和对应时间戳
            for i, frame in enumerate(frames):
                # 处理图像 - PIL图像转为base64
                buffer = io.BytesIO()
                frame["image"].save(buffer, format="JPEG", quality=80, optimize=True)
                img_bytes = buffer.getvalue()
                base64_data = base64.b64encode(img_bytes).decode("utf-8")
                img_data_uri = f"data:image/jpeg;base64,{base64_data}"
                
                # 添加图像 - 一个元素只有一个图像
                user_content.append({"image": img_data_uri})
                
                # 添加时间戳描述 - 一个元素只有一个文本
                user_content.append({"text": f"图像 {i+1}, 时间戳: {frame['position']}"})
            
            # 用户消息
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # 构建请求数据
            request_data = {
                "model": self.model_name,
                "input": {
                    "messages": messages
                },
                "parameters": {}
            }
            
            # 日志记录请求
            logger.info(f"发送请求到模型API, 使用模型: {self.model_name}, 帧数: {len(frames)}")
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 发送请求
            response = requests.post(
                self.base_url,
                json=request_data,
                headers=headers
            )
            
            # 处理错误
            if response.status_code != 200:
                logger.error(f"API返回错误: {response.status_code}, 响应: {response.text}")
                return f"API请求失败: HTTP {response.status_code}\n响应: {response.text}"
            
            # 解析响应
            result = response.json()
            
            try:
                # 根据DashScope API的实际响应格式提取文本
                if "output" in result and "choices" in result["output"] and len(result["output"]["choices"]) > 0:
                    choice = result["output"]["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        for content_item in choice["message"]["content"]:
                            if "text" in content_item:
                                return content_item["text"]
                    
                    # 如果找不到标准路径，尝试其他可能的路径
                    if "message" in choice and "content" in choice["message"] and isinstance(choice["message"]["content"], str):
                        return choice["message"]["content"]
                        
                    return "无法从API响应中提取文本内容"
                else:
                    logger.error(f"API返回了意外的响应格式: {result}")
                    return "API返回的数据格式有误，无法解析结果。"
            except Exception as e:
                logger.error(f"解析API响应时出错: {e}, 响应: {result}")
                # 尝试返回原始响应的字符串表示
                try:
                    return f"解析响应出错，原始响应：\n{json.dumps(result, ensure_ascii=False, indent=2)}"
                except:
                    return f"解析响应出错: {str(e)}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"请求API时出错: {e}")
            return f"请求错误: {str(e)}"
        except Exception as e:
            logger.error(f"分析视频帧时出错: {e}", exc_info=True)
            return f"处理错误: {str(e)}"

    # 在 model_api.py 的类中添加

    def debug_request(self, prompt: str, frames: List[Dict[str, Any]]) -> str:
        """生成调试信息但不发送API请求"""
        # 构建一个与实际请求相同的结构，但不包含实际的图片数据
        user_content = []
        user_content.append({"text": prompt})
        
        for i in range(len(frames)):
            user_content.append({"image": "[base64 image data]"})
            user_content.append({"text": f"图像 {i+1}, 时间戳: {frames[i]['position']}"})
        
        request_data = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {
                        "role": "system", 
                        "content": [{
                            "text": "系统提示内容..."
                        }]
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        
        return json.dumps(request_data, indent=2)


    def analyze_video_frames_mock(self, prompt: str, frames: List[Dict[str, Any]]) -> str:
        """模拟API调用（用于测试）
        
        Parameters:
            prompt: 提示词
            frames: 帧数据列表
            
        Returns:
            模拟的分析结果
        """
        # 模拟API延迟
        time.sleep(2)
        
        # 构造模拟响应
        frame_count = len(frames)
        response = f"## 视频分析报告\n\n"
        response += f"基于所提供的{frame_count}个视频帧，我进行了分析：\n\n"
        
        # 添加帧位置信息
        response += "### 关键时刻\n\n"
        for i, frame in enumerate(frames):
            response += f"- **{frame['position']}**: "
            response += f"在第{i+1}个关键帧中，可以观察到场景变化。\n"
        
        response += "\n### 总体分析\n\n"
        response += "这段视频展示了一系列连续的场景，视频质量良好，内容连贯。"
        response += "视频中的主要活动看起来集中在室内场景，有人物活动和互动。\n\n"
        
        response += "### 建议\n\n"
        response += "- 视频节奏适中，能够清晰传达主要信息\n"
        response += "- 光线和对比度适宜，画面清晰\n"
        response += "- 内容连贯且有明确的主题\n\n"
        
        return response