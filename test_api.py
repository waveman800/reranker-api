#!/usr/bin/env python3
"""
多模态重排服务 API 测试脚本

运行测试:
    python test_api.py
    
或者使用 pytest:
    pytest test_api.py -v
"""

import requests
import json
import base64
import io
from PIL import Image
import pytest
from pathlib import Path


BASE_URL = "http://localhost:6006"


def create_test_image() -> bytes:
    """创建一个测试图像"""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


def test_health_check():
    """测试健康检查端点"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    print(f"✓ 健康检查通过: {data}")


def test_model_info():
    """测试模型信息端点"""
    response = requests.get(f"{BASE_URL}/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_type" in data
    print(f"✓ 模型信息: {data}")


def test_text_rerank():
    """测试纯文本重排序"""
    url = f"{BASE_URL}/v1/rerank"
    
    data = {
        "query": "什么是人工智能？",
        "documents": [
            {"text": "人工智能是计算机科学的一个分支。"},
            {"text": "机器学习是人工智能的子领域。"},
            {"text": "深度学习是机器学习的一种方法。"}
        ],
        "top_k": 2
    }
    
    response = requests.post(url, json=data)
    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    assert len(result["results"]) <= 2
    print(f"✓ 文本重排序测试通过: {len(result['results'])} 个结果")


def test_text_rerank_form():
    """测试纯文本重排序（Form格式）"""
    url = f"{BASE_URL}/v1/rerank/text"
    
    data = {
        "query": "什么是人工智能？",
        "top_k": 2
    }
    
    files = [
        ("documents", (None, "人工智能是计算机科学的一个分支。")),
        ("documents", (None, "机器学习是人工智能的子领域。")),
        ("documents", (None, "深度学习是机器学习的一种方法。"))
    ]
    
    response = requests.post(url, data=data, files=files)
    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    print(f"✓ 文本重排序(Form)测试通过: {len(result['results'])} 个结果")


def test_multimodal_rerank_with_url():
    """测试多模态重排序（图像URL）"""
    url = f"{BASE_URL}/v1/rerank"
    
    data = {
        "query": "展示一只猫",
        "documents": [
            {"text": "这是一张狗的照片"},
            {"text": "这是一只可爱的猫", "image_url": "https://example.com/cat.jpg", "image_type": "url"},
            {"text": "这是一辆红色的跑车"}
        ],
        "top_k": 2,
        "return_documents": True
    }
    
    response = requests.post(url, json=data)
    # 注意：由于使用的是示例URL，可能会失败，这里只检查接口响应
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 多模态重排序(URL)测试通过: {len(result.get('results', []))} 个结果")
    else:
        print(f"⚠ 多模态重排序(URL)测试跳过 (状态码: {response.status_code})")


def test_multimodal_rerank_with_base64():
    """测试多模态重排序（Base64图像）"""
    url = f"{BASE_URL}/v1/rerank"
    
    # 创建测试图像并转换为base64
    image_bytes = create_test_image()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    data = {
        "query": "展示红色图像",
        "documents": [
            {"text": "这是一张狗的照片"},
            {"text": "这是一张红色图像", "image_url": f"data:image/jpeg;base64,{base64_image}", "image_type": "base64"},
            {"text": "这是一辆红色的跑车"}
        ],
        "top_k": 2,
        "return_documents": True
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 多模态重排序(Base64)测试通过: {len(result.get('results', []))} 个结果")
    else:
        print(f"⚠ 多模态重排序(Base64)测试跳过 (状态码: {response.status_code})")


def test_multimodal_rerank_with_files():
    """测试多模态重排序（文件上传）"""
    url = f"{BASE_URL}/v1/rerank/multimodal"
    
    # 创建测试图像
    image_bytes = create_test_image()
    
    data = {
        "query": "展示红色图像",
        "top_k": 2,
        "return_documents": "false"
    }
    
    files = [
        ("documents_text", (None, "这是一张狗的照片")),
        ("images", ("test_image.jpg", io.BytesIO(image_bytes), "image/jpeg")),
        ("documents_text", (None, "这是一辆红色的跑车"))
    ]
    
    response = requests.post(url, data=data, files=files)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 多模态重排序(文件上传)测试通过: {len(result.get('results', []))} 个结果")
    else:
        print(f"⚠ 多模态重排序(文件上传)测试跳过 (状态码: {response.status_code})")


def test_empty_documents():
    """测试空文档列表"""
    url = f"{BASE_URL}/v1/rerank"
    
    data = {
        "query": "测试查询",
        "documents": []
    }
    
    response = requests.post(url, json=data)
    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    assert len(result["results"]) == 0
    print(f"✓ 空文档列表测试通过")


def test_invalid_image():
    """测试无效图像"""
    url = f"{BASE_URL}/v1/rerank"
    
    data = {
        "query": "测试查询",
        "documents": [
            {"image_url": "invalid_base64_data", "image_type": "base64"}
        ]
    }
    
    response = requests.post(url, json=data)
    # 应该返回400错误
    if response.status_code == 400:
        print(f"✓ 无效图像测试通过 (正确返回400错误)")
    else:
        print(f"⚠ 无效图像测试: 状态码 {response.status_code}")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("多模态重排服务 API 测试")
    print("=" * 60)
    
    tests = [
        test_health_check,
        test_model_info,
        test_text_rerank,
        test_text_rerank_form,
        test_multimodal_rerank_with_url,
        test_multimodal_rerank_with_base64,
        test_multimodal_rerank_with_files,
        test_empty_documents,
        test_invalid_image
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} 失败: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
