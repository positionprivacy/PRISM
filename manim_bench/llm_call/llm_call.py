#!/usr/bin/env python3
"""
LLM API Client using OpenAI with configuration from config.json
"""

import base64
import time
from openai import OpenAI
import json
import socket
import ssl
import re
import os
import argparse
from typing import List, Dict, Any, Union, Tuple, Optional
from PIL import Image


class LLMAPIClient:
    """LLM API client that handles configuration and API operations"""

    BUSY_MESSAGE = "服务器繁忙，请稍后再试吧"
    
    def __init__(self, config_path="config.json", model_override=None):
        """Initialize the LLM API Client with configuration from JSON file"""
        self.config_path = config_path
        self.config = self._load_config()
        self.last_response_usage = None
        
        # 从配置获取设置
        llm_settings = self.config.get('llm_settings', {})
        self.api_key = self.config['llm_key']
        # 优先使用覆盖的模型名
        self.model = model_override if model_override else llm_settings.get('model', 'gpt-5-chat')
        self.base_url = llm_settings.get('base_url', 'https://yeysai.com/v1/')
        self.max_tokens = llm_settings.get('max_tokens', 3200)
        self.temperature = llm_settings.get('temperature', 1)
        self.max_retries = llm_settings.get('max_retries', 3)
        self.timeout = llm_settings.get('timeout', 1200)
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_file_path = os.path.join(script_dir, self.config_path)
            
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if not config.get('llm_key') or config['llm_key'] == "your_llm_api_key_here":
                raise ValueError("Please set your llm_key in config.json")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {self.config_path}")
    
    def encode_image(self, image_path: str) -> str:
        """将图片编码为base64格式"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"图片编码失败: {str(e)}")
    
    def extract_images_from_text(self, text: str) -> List[str]:
        """从文本中提取图片路径，格式为 ![](xx/xxx.jpg)，但忽略被引号包裹的情况"""
        # 先找出所有的图片引用
        pattern = r"!\[\]\((.+?)\)"
        matches = []
        
        # 对每个匹配项，检查它是否被引号包裹
        for match in re.finditer(pattern, text):
            start = match.start()
            end = match.end()
            
            # 检查这个匹配是否被引号包裹
            # 向前找最近的引号
            prev_quote = text.rfind("'", 0, start)
            prev_quote2 = text.rfind("`", 0, start)
            prev_quote = max(prev_quote, prev_quote2)
            
            # 向后找最近的引号
            next_quote = text.find("'", end)
            next_quote2 = text.find("`", end)
            if next_quote == -1:
                next_quote = len(text)
            if next_quote2 == -1:
                next_quote2 = len(text)
            next_quote = min(next_quote, next_quote2)
            
            # 如果这个图片引用不在引号内，就添加到结果中
            if prev_quote == -1 or next_quote == -1 or not (prev_quote < start and end < next_quote):
                matches.append(match.group(1))
                
        return matches
    
    def get_image_size(self, image_path: str) -> Tuple[int, int]:
        """获取图片的尺寸 (width, height)"""
        with Image.open(image_path) as img:
            return img.width, img.height
    
    def get_mime_type(self, file_path: str) -> str:
        """根据文件扩展名获取MIME类型"""
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml'
        }
        return mime_types.get(ext, 'application/octet-stream')
    
    def load_prompt_template(self, template_name: str) -> str:
        """
        加载prompt模板文件
        
        Args:
            template_name: 模板文件名（不包含.txt扩展名）
        
        Returns:
            模板内容字符串
        """
        try:
            # 获取当前脚本目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(script_dir, "prompt_templates", f"{template_name}.txt")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file {template_name}.txt not found in prompt_templates/ directory")
        except Exception as e:
            raise Exception(f"Error loading template: {str(e)}")
    
    def create_noter_prompt(self, keyword: str) -> str:
        """
        创建用于Noter任务的prompt
        
        Args:
            keyword: 机器学习关键词
        
        Returns:
            完整的prompt字符串
        """
        template = self.load_prompt_template("Noter")
        return template.replace("{keyword}", keyword)

    def resolve_image_path(self, img_path: str, base_path: Optional[str] = None) -> str:
        """
        解析图片路径，将相对路径转换为基于base_path的绝对路径
        
        Args:
            img_path: 原始图片路径
            base_path: 基准路径（通常是markdown文件所在目录）
        
        Returns:
            解析后的图片路径
        """
        if os.path.isabs(img_path):
            # 如果已经是绝对路径，直接返回
            return img_path
        
        if base_path:
            # 如果提供了base_path，相对于base_path解析
            resolved_path = os.path.join(base_path, img_path)
            return os.path.abspath(resolved_path)
        else:
            # 如果没有base_path，相对于当前工作目录
            return os.path.abspath(img_path)

    def call_api_with_text_and_images(self, text: str, base_path: Optional[str] = None) -> str:
        """
        处理文本中的图片引用并调用API
        
        Args:
            text: 要处理的文本
            base_path: 图片路径的基准目录（通常是markdown文件所在目录）
        """
        # 提取图片路径
        image_paths = self.extract_images_from_text(text)
    
        modified_text = text
        offset = 0  # 由于插入新字符，原始索引会发生偏移

        for match in re.finditer(r"!\[\]\((.+?)\)", text):
            img_path = match.group(1)
            try:
                # 解析图片路径
                resolved_img_path = self.resolve_image_path(img_path, base_path)
                width, height = self.get_image_size(resolved_img_path)
                size_str = f"（尺寸：{width}×{height}）"
                insert_pos = match.end() + offset
                modified_text = modified_text[:insert_pos] + size_str + modified_text[insert_pos:]
                offset += len(size_str)  # 更新偏移量
            except Exception as e:
                print(f"读取图片尺寸失败 {img_path}: {e}")

        # 准备消息内容
        content = []
        
        # 保留原始文本中的图片引用
        content.append({"type": "text", "text": modified_text})

        # 添加图片内容
        for img_path in image_paths:
            try:
                # 解析图片路径
                resolved_img_path = self.resolve_image_path(img_path, base_path)
                base64_image = self.encode_image(resolved_img_path)
                mime_type = self.get_mime_type(resolved_img_path)
                image_data_url = f"data:{mime_type};base64,{base64_image}"
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                })
            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {str(e)}")
        
        # 调用API
        return self._call_api(content)
    
    # def call_api_with_text(self, text: str) -> str:
    #     """简单的纯文本API调用，不处理图片"""
    #     content = [
    #         {
    #             "type": "text",
    #             "text": text
    #         }
    #     ]
        
    #     # 调用API
    #     return self._call_api(content)

    def call_api_with_text(self, text: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """简单的纯文本API调用，不处理图片"""
        content = [
            {
                "type": "text",
                "text": text
            }
        ]
        
        # 调用API，传入自定义参数
        return self._call_api(content, max_tokens=max_tokens, temperature=temperature)
    
    def call_api_with_text_stream(self, text: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None):
        """
        简单的纯文本流式API调用，不处理图片
        优化版本：立即返回第一个chunk，最小化延迟
        
        Args:
            text: 要发送的文本
            max_tokens: 最大token数（可选，默认使用实例配置）
            temperature: 温度参数（可选，默认使用实例配置）
        
        Yields:
            str: 每次返回的文本片段
        """
        content = [
            {
                "type": "text",
                "text": text
            }
        ]
        
        # 调用流式API，传入自定义参数
        yield from self._call_api_stream(content, max_tokens=max_tokens, temperature=temperature)
    
    def generate_course_notes(self, keyword: str) -> str:
        """
        根据关键词生成机器学习课程讲义大纲
        
        Args:
            keyword: 机器学习关键词（如"KNN", "SVM"等）
        
        Returns:
            生成的课程讲义大纲
        """
        prompt = self.create_noter_prompt(keyword)
        return self.call_api_with_text(prompt)
    
    def create_script_writer_prompt(self, keyword: str, search_results: str) -> str:
        """
        创建用于Script Writer任务的prompt
        
        Args:
            keyword: 机器学习关键词
            search_results: 检索得到的信息
        
        Returns:
            完整的prompt字符串
        """
        template = self.load_prompt_template("Script_Writer")
        return template.replace("{keyword}", keyword).replace("{search_results}", search_results)
    
    def create_chapter_writer_prompt(self, chapter_topic: str, search_results: str) -> str:
        """
        创建用于Chapter Writer任务的prompt
        
        Args:
            chapter_topic: 章节主题
            search_results: 检索得到的信息
        
        Returns:
            完整的prompt字符串
        """
        template = self.load_prompt_template("Chapter_Writer")
        return template.replace("{chapter_topic}", chapter_topic).replace("{search_results}", search_results)
    
    def create_brain_prompt(self, section_content: str) -> str:
        """
        创建用于Brain任务的prompt（分页处理）
        
        Args:
            section_content: 章节内容
        
        Returns:
            完整的prompt字符串
        """
        template = self.load_prompt_template("Brain")
        return template.replace("{section_content}", section_content)
    
    def generate_teaching_script(self, keyword: str, search_results: str) -> str:
        """
        根据关键词和检索信息生成完整的教学讲义
        
        Args:
            keyword: 机器学习关键词（如"KNN", "SVM"等）
            search_results: 检索得到的相关信息
        
        Returns:
            生成的完整教学讲义
        """
        prompt = self.create_script_writer_prompt(keyword, search_results)
        return self.call_api_with_text(prompt)
    
    def generate_chapter_script(self, chapter_topic: str, search_results: str) -> str:
        """
        根据章节主题和检索信息生成该章节的详细讲义
        
        Args:
            chapter_topic: 章节主题
            search_results: 检索得到的相关信息
        
        Returns:
            生成的章节详细讲义
        """
        prompt = self.create_chapter_writer_prompt(chapter_topic, search_results)
        return self.call_api_with_text(prompt)
    
    def generate_paginated_section(self, section_content: str) -> str:
        """
        对章节内容进行分页处理
        
        Args:
            section_content: 原始章节内容
        
        Returns:
            添加了分页标记的章节内容
        """
        prompt = self.create_brain_prompt(section_content)
        return self.call_api_with_text(prompt)
    
    def _call_api(
        self,
        content: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """发送API请求并处理响应"""
        busy_message = self.BUSY_MESSAGE
        actual_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        actual_temperature = temperature if temperature is not None else self.temperature
        retry_count = 0
        response_content = None
        self.last_response_usage = None

        while retry_count < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=actual_max_tokens,
                    temperature=actual_temperature
                )
                self.last_response_usage = self._extract_usage(response)
                
                if response.choices and response.choices[0].message:
                    response_content = response.choices[0].message.content
                else:
                    response_content = busy_message
                break  # 成功，跳出重试循环

            except Exception as e:
                retry_count += 1
                print(f"API调用错误 (尝试 {retry_count}/{self.max_retries}): {e}")
                if retry_count >= self.max_retries:
                    response_content = busy_message
                    break
                print(f"等待 {5 * retry_count} 秒后重试...")  # 简单的退避策略
                time.sleep(5 * retry_count)

        return response_content if response_content else busy_message

    def _extract_usage(self, response) -> Optional[Dict[str, Any]]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        def read_field(obj, key, default=0):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        completion_tokens_details = read_field(usage, "completion_tokens_details", None)
        prompt_tokens_details = read_field(usage, "prompt_tokens_details", None)

        payload = {
            "prompt_tokens": int(read_field(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(read_field(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(read_field(usage, "total_tokens", 0) or 0),
            "reasoning_tokens": int(read_field(completion_tokens_details, "reasoning_tokens", 0) or 0),
            "cached_tokens": int(read_field(prompt_tokens_details, "cached_tokens", 0) or 0),
        }
        return payload

    def _call_api_stream(self, content: List[Dict[str, Any]], max_tokens: Optional[int] = None, temperature: Optional[float] = None):
        """
        发送流式API请求并逐步返回响应（优化版本：最小化延迟）
        
        Args:
            content: 消息内容
            max_tokens: 最大token数（可选，默认使用实例配置）
            temperature: 温度参数（可选，默认使用实例配置）
        
        Yields:
            str: 每次返回的文本片段
        """
        busy_message = self.BUSY_MESSAGE
        # 使用传入的参数，如果没有则使用实例配置
        actual_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        actual_temperature = temperature if temperature is not None else self.temperature

        retry_count = 0

        while retry_count < self.max_retries:
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=actual_max_tokens,
                    temperature=actual_temperature,
                    stream=True  # 启用流式输出
                )
                
                # 立即yield每个chunk，最小化延迟
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content
                
                # 成功完成，退出重试循环
                return
                        
            except Exception as e:
                retry_count += 1
                print(f"流式API调用错误 (尝试 {retry_count}/{self.max_retries}): {e}")
                
                if retry_count >= self.max_retries:
                    # 达到最大重试次数：统一返回忙碌提示，不暴露底层错误细节
                    yield busy_message
                    return
                
                # 重试前短暂等待（减少等待时间以提高响应速度）
                wait_time = min(2 * retry_count, 5)  # 最多等待5秒
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)


def process_text_with_images(text: str, config_path: str = "config.json", base_path: Optional[str] = None) -> str:
    """
    处理包含图片的文本
    
    Args:
        text: 要处理的文本
        config_path: 配置文件路径
        base_path: 图片路径的基准目录（通常是markdown文件所在目录）
    """
    client = LLMAPIClient(config_path=config_path)
    return client.call_api_with_text_and_images(text, base_path)

def process_text(text: str, config_path: str = "config.json") -> str:
    """简单的纯文本处理函数"""
    client = LLMAPIClient(config_path=config_path)
    return client.call_api_with_text(text)

def generate_course_notes(keyword: str, config_path: str = "config.json") -> str:
    """
    根据关键词生成机器学习课程讲义大纲的便利函数
    
    Args:
        keyword: 机器学习关键词（如"KNN", "SVM"等）
        config_path: 配置文件路径
    
    Returns:
        生成的课程讲义大纲
    """
    client = LLMAPIClient(config_path=config_path)
    return client.generate_course_notes(keyword)

def generate_teaching_script(keyword: str, search_results: str, config_path: str = "config.json") -> str:
    """
    根据关键词和检索信息生成完整教学讲义的便利函数
    
    Args:
        keyword: 机器学习关键词（如"KNN", "SVM"等）
        search_results: 检索得到的相关信息
        config_path: 配置文件路径
    
    Returns:
        生成的完整教学讲义
    """
    client = LLMAPIClient(config_path=config_path)
    return client.generate_teaching_script(keyword, search_results)


def main():
    """Main function that handles command line arguments and performs LLM API calls"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="LLM API Client - send text to LLM and get response",
        epilog="Examples:\n"
               "  python llm_api.py \"解释一下机器学习的基本概念\"\n"
               "  python llm_api.py \"KNN\" --noter\n"
               "  python llm_api.py \"SVM\" --noter",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "text", 
        nargs="?",  # Makes the text optional
        default="Hello, how are you?",
        help="Text to send to LLM or keyword for course notes (default: 'Hello, how are you?')"
    )
    parser.add_argument(
        "--noter", 
        action="store_true",
        help="Generate machine learning course notes outline based on keyword"
    )
    parser.add_argument(
        "--with-images", 
        action="store_true",
        help="Process images in text (look for ![](path) patterns)"
    )
    parser.add_argument(
        "--base-path", 
        type=str,
        default=None,
        help="Base path for resolving relative image paths"
    )
    parser.add_argument(
        "--config", 
        type=str,
        default="config.json",
        help="Path to config file (default: config.json)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Initialize the LLM API client
        client = LLMAPIClient(config_path=args.config)
        
        # Process based on mode
        if args.noter:
            # Course notes generation mode
            result = client.generate_course_notes(args.text)
            print(result)
        else:
            # Normal text processing mode
            print(f"输入文本: '{args.text}'")
            print("=" * 50)
            
            # Process text based on arguments
            if args.with_images:
                print("处理模式: 文本+图片")
                result = client.call_api_with_text_and_images(args.text, base_path=args.base_path)
            else:
                print("处理模式: 纯文本")
                result = client.call_api_with_text(args.text)
            
            print("\n=== LLM 响应 ===")
            print(result)
            
    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保:")
        print("1. 安装必要的包: pip install openai pillow")
        print("2. 在 config.json 中设置你的 llm_key")


if __name__ == "__main__":
    main()
