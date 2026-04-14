# 多模态重排服务 (Multimodal Reranker API)

基于 Qwen3-VL-Reranker-2B 模型的多模态重排序 API 服务，支持对文本和图像文档进行相关性重排序。本服务使用 Docker 容器化部署，支持 GPU 加速和自动混合精度计算。

## ✨ 功能特点

- 🚀 支持批量文档重排序（文本 + 图像）
- 🖼️ 支持多模态输入（图像URL、Base64编码图像、文件上传）
- ⚡ 支持 GPU 加速（NVIDIA GPU 推荐）
- 🔒 支持访问令牌认证
- 📊 自动混合精度计算（BF16/FP16/FP32）
- 🐳 Docker 容器化部署
- 🔍 提供 RESTful API 接口
- 💪 支持多工作进程
- 🩺 健康检查接口
- 🔄 向后兼容纯文本重排序 API

## 🚀 快速开始

### 1. 环境准备

- Docker 19.03+
- Docker Compose 1.28+
- NVIDIA Container Toolkit（如需 GPU 支持）
- 至少 16GB 显存（推荐 24GB+ 用于完整模型）

### 2. 获取代码

```bash
git clone <your-repo-url>
cd reranker-api
```

### 3. 下载模型

下载 Qwen3-VL-Reranker-2B 模型到 `models/` 目录：

```bash
# 从 Hugging Face 下载
huggingface-cli download Qwen/Qwen3-VL-Reranker-2B --local-dir ./models/Qwen3-VL-Reranker-2B

# 或者从 ModelScope 下载
modelscope download --model qwen/Qwen3-VL-Reranker-2B --local_dir ./models/Qwen3-VL-Reranker-2B
```

### 4. 配置

1. 复制示例配置文件：

```bash
cp .env.example .env
```

2. 修改 `.env` 文件中的配置项：

```ini
# 服务器配置
HOST=0.0.0.0
PORT=6006
WORKERS=1

# 模型配置
MODEL_NAME_OR_PATH=./models/Qwen3-VL-Reranker-2B
MODEL_TYPE=multimodal  # text / multimodal / auto
MODEL_DEVICE=cuda
MODEL_PRECISION=bf16

# 性能配置
MAX_SEQ_LENGTH=512
BATCH_SIZE=4
MAX_MEMORY_PERCENTAGE=80

# 访问令牌（可选）
# ACCESS_TOKEN=your-secret-token
```

### 5. 启动服务

使用 Docker Compose 启动服务：

```bash
docker-compose up -d --build
```

### 6. 验证服务

检查服务健康状态：

```bash
curl http://localhost:6006/health
```

正常响应：
```json
{
  "status": "healthy",
  "model_type": "multimodal",
  "device": "cuda"
}
```

## 📚 API 文档

### 1. 多模态重排序（JSON格式）

**端点**

```
POST /v1/rerank
```

**请求头**

```
Content-Type: application/json
# 如果设置了 ACCESS_TOKEN，需要提供以下任一认证方式：
# 1. 请求头: Authorization: Bearer <your_access_token>
# 2. URL 参数: ?access_token=<your_access_token>
```

**请求体**

```json
{
  "query": "查询文本",
  "documents": [
    {
      "text": "文本文档内容"
    },
    {
      "image_url": "https://example.com/image.jpg",
      "image_type": "url"
    },
    {
      "text": "图文混合文档",
      "image_url": "data:image/jpeg;base64,/9j/4AAQ...",
      "image_type": "base64"
    }
  ],
  "top_k": 3,
  "return_documents": true
}
```

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| query | string | 是 | 查询文本 |
| documents | array | 是 | 文档列表，每个文档可以是文本、图像或两者都有 |
| documents[].text | string | 否 | 文档文本内容 |
| documents[].image_url | string | 否 | 图像URL或base64编码 |
| documents[].image_type | string | 否 | 图像类型: url 或 base64 |
| top_k | integer | 否 | 返回最相关的 k 个结果，默认返回所有 |
| return_documents | boolean | 否 | 是否返回原始文档内容 |

**示例请求**

```bash
curl -X 'POST' \
  'http://localhost:6006/v1/rerank' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "展示一只猫的图片",
    "documents": [
      {"text": "这是一张狗的照片"},
      {"image_url": "https://example.com/cat.jpg", "image_type": "url"},
      {"text": "可爱的小猫咪", "image_url": "data:image/jpeg;base64,/9j/4AAQ...", "image_type": "base64"}
    ],
    "top_k": 2,
    "return_documents": true
  }'
```

**成功响应**

```json
{
  "results": [
    {
      "index": 1,
      "score": 0.95,
      "document": {
        "image_url": "https://example.com/cat.jpg",
        "image_type": "url"
      }
    },
    {
      "index": 2,
      "score": 0.88,
      "document": {
        "text": "可爱的小猫咪",
        "image_url": "data:image/jpeg;base64,/9j/4AAQ...",
        "image_type": "base64"
      }
    }
  ],
  "model_type": "multimodal"
}
```

### 2. 纯文本重排序（向后兼容）

**端点**

```
POST /v1/rerank/text
```

**请求体（Form格式）**

```
query=查询文本
documents=文档1
documents=文档2
documents=文档3
top_k=2
```

**示例请求**

```bash
curl -X 'POST' \
  'http://localhost:6006/v1/rerank/text' \
  -F 'query=什么是人工智能？' \
  -F 'documents=人工智能是计算机科学的一个分支' \
  -F 'documents=机器学习是人工智能的子领域' \
  -F 'top_k=2'
```

### 3. 文件上传多模态重排序

**端点**

```
POST /v1/rerank/multimodal
```

**请求体（Multipart Form格式）**

```
query=查询文本
documents_text=文本文档1
documents_text=文本文档2
images=@/path/to/image1.jpg
images=@/path/to/image2.png
top_k=2
return_documents=true
```

**示例请求**

```bash
curl -X 'POST' \
  'http://localhost:6006/v1/rerank/multimodal' \
  -F 'query=展示一只猫的图片' \
  -F 'documents_text=这是一张狗的照片' \
  -F 'images=@/path/to/cat.jpg' \
  -F 'top_k=2'
```

### 4. 健康检查

**端点**

```
GET /health
```

**响应**

```json
{
  "status": "healthy",
  "model_type": "multimodal",
  "device": "cuda"
}
```

### 5. 模型信息

**端点**

```
GET /v1/model/info
```

**响应**

```json
{
  "model_name": "/app/models/Qwen3-VL-Reranker-2B",
  "model_type": "multimodal",
  "device": "cuda",
  "precision": "bf16",
  "max_length": 512,
  "batch_size": 4
}
```

## 🐳 Docker 构建

### 构建镜像

```bash
docker-compose build --no-cache
```

### 查看日志

```bash
docker-compose logs -f
```

### 停止服务

```bash
docker-compose down
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `6006` | 服务监听端口 |
| `MODEL_NAME_OR_PATH` | `./models/Qwen3-VL-Reranker-2B` | 模型路径 |
| `MODEL_TYPE` | `multimodal` | 模型类型: text/multimodal/auto |
| `MODEL_DEVICE` | `cuda` | 运行设备 (`cuda`/`cpu`) |
| `MODEL_PRECISION` | `bf16` | 模型精度 (`bf16`/`fp16`/`fp32`) |
| `MAX_MEMORY_PERCENTAGE` | `80` | 显存使用百分比 |
| `BATCH_SIZE` | `4` | 批处理大小 |
| `MAX_SEQ_LENGTH` | `512` | 最大序列长度 |
| `ACCESS_TOKEN` | `None` | 访问令牌（可选） |

## 📄 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 🔧 开发

### 环境设置

1. 创建并激活虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

### 本地运行

```bash
uvicorn app:app --host 0.0.0.0 --port 6006 --reload
```

## 💡 使用示例

### Python 客户端示例

```python
import requests
import base64

def rerank_multimodal(query, documents, top_k=None, api_key=None):
    """
    多模态重排序
    
    documents 格式:
    [
        {"text": "文本文档"},
        {"image_url": "https://example.com/image.jpg", "image_type": "url"},
        {"text": "图文混合", "image_url": "base64encoded...", "image_type": "base64"}
    ]
    """
    url = "http://localhost:6006/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else ""
    }
    
    data = {
        "query": query,
        "documents": documents,
        "top_k": top_k,
        "return_documents": True
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 使用示例
documents = [
    {"text": "这是一张狗的照片"},
    {"image_url": "https://example.com/cat.jpg", "image_type": "url"},
    {"text": "可爱的小猫咪", "image_url": "data:image/jpeg;base64,/9j/4AAQ...", "image_type": "base64"}
]

results = rerank_multimodal(
    query="展示一只猫的图片",
    documents=documents,
    top_k=2
)
print(results)
```

### 图像文件上传示例

```python
import requests

def rerank_with_files(query, text_docs=None, image_paths=None, top_k=None):
    """使用文件上传进行多模态重排序"""
    url = "http://localhost:6006/v1/rerank/multimodal"
    
    data = {"query": query}
    if top_k:
        data["top_k"] = top_k
    
    files = []
    if text_docs:
        for doc in text_docs:
            files.append(("documents_text", (None, doc)))
    
    if image_paths:
        for path in image_paths:
            files.append(("images", open(path, "rb")))
    
    response = requests.post(url, data=data, files=files)
    return response.json()

# 使用示例
results = rerank_with_files(
    query="展示一只猫的图片",
    text_docs=["这是一张狗的照片", "可爱的小猫咪"],
    image_paths=["/path/to/cat.jpg", "/path/to/dog.jpg"],
    top_k=2
)
print(results)
```

## ⚡ 性能优化

1. **调整批处理大小**：
   - 多模态模型建议设置较小的 batch_size（如4）
   - 在 `.env` 中修改 `BATCH_SIZE` 参数

2. **调整工作进程数**：
   - 对于GPU服务，建议保持 `WORKERS=1` 以避免显存竞争
   - 在 `.env` 中修改 `WORKERS` 参数

3. **精度设置**：
   - 使用 `bf16` 或 `fp16` 可以减少显存使用
   - 在 `.env` 中修改 `MODEL_PRECISION`

4. **显存优化**：
   - 调整 `MAX_MEMORY_PERCENTAGE` 限制显存使用
   - 减小 `MAX_SEQ_LENGTH` 以降低内存占用

## ❓ 常见问题

### 1. 显存不足

- 减小 `BATCH_SIZE`（建议多模态设置为2-4）
- 减小 `MAX_MEMORY_PERCENTAGE`
- 使用更小的模型（如 Qwen3-VL-Reranker-2B）
- 使用 CPU 模式（性能较低）

### 2. 请求超时

- 增加 `TIMEOUT` 参数
- 减少批处理大小
- 优化图像尺寸

### 3. 模型加载失败

- 确保模型文件已正确下载到 `models/` 目录
- 检查模型路径配置是否正确
- 查看日志获取详细错误信息

### 4. 图像处理错误

- 确保图像格式支持（JPG, PNG, WebP等）
- 检查图像URL是否可访问
- 验证base64编码是否正确

## 📞 支持

如有问题或建议，请提交 Issue 或 Pull Request。
