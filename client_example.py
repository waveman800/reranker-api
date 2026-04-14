#!/usr/bin/env python3
"""
多模态重排服务客户端示例

支持以下功能：
1. 纯文本重排序
2. 图像URL重排序
3. Base64编码图像重排序
4. 文件上传重排序
"""

import requests
import base64
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class MultimodalRerankerClient:
    """多模态重排服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:6006", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_documents: bool = False
    ) -> Dict[str, Any]:
        """
        多模态重排序（JSON格式）
        
        Args:
            query: 查询文本
            documents: 文档列表，每个文档可以是:
                - {"text": "文本文档内容"}
                - {"image_url": "https://example.com/image.jpg", "image_type": "url"}
                - {"text": "图文混合", "image_url": "data:image/jpeg;base64,...", "image_type": "base64"}
            top_k: 返回最相关的k个结果
            return_documents: 是否返回原始文档内容
        
        Returns:
            重排序结果
        """
        url = f"{self.base_url}/v1/rerank"
        
        data = {
            "query": query,
            "documents": documents,
            "return_documents": return_documents
        }
        if top_k is not None:
            data["top_k"] = top_k
        
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def rerank_text(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        纯文本重排序（向后兼容）
        
        Args:
            query: 查询文本
            documents: 文本文档列表
            top_k: 返回最相关的k个结果
        
        Returns:
            重排序结果
        """
        url = f"{self.base_url}/v1/rerank/text"
        
        data = {"query": query}
        if top_k is not None:
            data["top_k"] = top_k
        
        # 使用multipart form格式
        files = [("documents", (None, doc)) for doc in documents]
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()
        return response.json()
    
    def rerank_with_files(
        self,
        query: str,
        text_docs: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        return_documents: bool = False
    ) -> Dict[str, Any]:
        """
        使用文件上传进行多模态重排序
        
        Args:
            query: 查询文本
            text_docs: 文本文档列表
            image_paths: 图像文件路径列表
            top_k: 返回最相关的k个结果
            return_documents: 是否返回原始文档内容
        
        Returns:
            重排序结果
        """
        url = f"{self.base_url}/v1/rerank/multimodal"
        
        data = {
            "query": query,
            "return_documents": str(return_documents).lower()
        }
        if top_k is not None:
            data["top_k"] = top_k
        
        files = []
        
        # 添加文本文档
        if text_docs:
            for doc in text_docs:
                files.append(("documents_text", (None, doc)))
        
        # 添加图像文件
        if image_paths:
            for path in image_paths:
                path_obj = Path(path)
                if not path_obj.exists():
                    raise FileNotFoundError(f"图像文件不存在: {path}")
                files.append(("images", (path_obj.name, open(path, "rb"))))
        
        try:
            response = requests.post(url, data=data, files=files)
            response.raise_for_status()
            return response.json()
        finally:
            # 关闭所有打开的文件
            for _, file_tuple in files:
                if isinstance(file_tuple, tuple) and hasattr(file_tuple[1], 'close'):
                    file_tuple[1].close()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        url = f"{self.base_url}/v1/model/info"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """将图像文件转换为base64编码"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    @staticmethod
    def create_image_document(image_path: str, text: Optional[str] = None) -> Dict[str, str]:
        """创建图像文档"""
        base64_image = MultimodalRerankerClient.image_to_base64(image_path)
        
        # 检测图像类型
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }.get(ext, 'image/jpeg')
        
        doc = {
            "image_url": f"data:{mime_type};base64,{base64_image}",
            "image_type": "base64"
        }
        if text:
            doc["text"] = text
        return doc


def demo_text_rerank():
    """纯文本重排序示例"""
    print("=" * 50)
    print("纯文本重排序示例")
    print("=" * 50)
    
    client = MultimodalRerankerClient()
    
    query = "什么是人工智能？"
    documents = [
        "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统。",
        "机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。",
        "深度学习是机器学习的一个子集，使用神经网络进行学习。",
        "今天天气真好，适合去公园散步。"
    ]
    
    try:
        results = client.rerank_text(query, documents, top_k=3)
        print(f"查询: {query}")
        print(f"结果:")
        for result in results.get("results", []):
            idx = result["index"]
            score = result["score"]
            print(f"  [{idx}] 分数: {score:.4f} - {documents[idx][:50]}...")
    except Exception as e:
        print(f"错误: {e}")


def demo_multimodal_rerank():
    """多模态重排序示例（使用模拟数据）"""
    print("\n" + "=" * 50)
    print("多模态重排序示例")
    print("=" * 50)
    
    client = MultimodalRerankerClient()
    
    query = "展示一只可爱的猫"
    
    # 模拟文档（实际使用时需要提供真实的图像URL或base64编码）
    documents = [
        {"text": "这是一张狗的照片，狗在草地上玩耍"},
        {"text": "可爱的小猫咪在阳光下睡觉", "image_url": "https://example.com/cat.jpg", "image_type": "url"},
        {"text": "这是一辆红色的跑车"},
    ]
    
    print(f"查询: {query}")
    print(f"文档数量: {len(documents)}")
    print("注意: 此示例使用模拟数据，实际使用时需要提供真实的图像")
    
    try:
        # 如果服务可用，尝试调用
        health = client.health_check()
        print(f"服务状态: {health}")
        
        # 实际调用（需要真实图像数据）
        # results = client.rerank(query, documents, top_k=2, return_documents=True)
        # print(f"结果: {json.dumps(results, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"服务未启动或发生错误: {e}")


def demo_file_upload():
    """文件上传重排序示例"""
    print("\n" + "=" * 50)
    print("文件上传重排序示例")
    print("=" * 50)
    
    client = MultimodalRerankerClient()
    
    query = "展示一只可爱的动物"
    
    # 示例文本文档
    text_docs = [
        "这是一张狗的照片",
        "可爱的小猫咪"
    ]
    
    print(f"查询: {query}")
    print(f"文本文档: {text_docs}")
    print("注意: 此示例需要真实的图像文件路径")
    
    # 示例图像路径（需要替换为实际路径）
    # image_paths = ["/path/to/cat.jpg", "/path/to/dog.jpg"]
    # results = client.rerank_with_files(query, text_docs, image_paths, top_k=2)


def demo_base64_image():
    """Base64编码图像重排序示例"""
    print("\n" + "=" * 50)
    print("Base64编码图像重排序示例")
    print("=" * 50)
    
    client = MultimodalRerankerClient()
    
    query = "展示一只可爱的猫"
    
    # 示例：将图像转换为base64并创建文档
    # image_path = "/path/to/cat.jpg"
    # image_doc = client.create_image_document(image_path, text="可爱的小猫咪")
    
    documents = [
        {"text": "这是一张狗的照片"},
        # image_doc,  # 包含图像的文档
        {"text": "这是一辆红色的跑车"}
    ]
    
    print(f"查询: {query}")
    print(f"文档数量: {len(documents)}")
    print("注意: 此示例需要真实的图像文件路径")
    
    # 实际调用
    # results = client.rerank(query, documents, top_k=2, return_documents=True)


if __name__ == "__main__":
    # 运行示例
    demo_text_rerank()
    demo_multimodal_rerank()
    demo_file_upload()
    demo_base64_image()
    
    print("\n" + "=" * 50)
    print("示例运行完成")
    print("=" * 50)
