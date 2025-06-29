# Reranker API 服务

基于 Qwen3-Reranker-4B 模型的重排序 API 服务，支持对文档进行相关性重排序。本服务使用 Docker 容器化部署，支持 GPU 加速和自动混合精度计算。

## ✨ 功能特点

- 🚀 支持批量文档重排序
- ⚡ 支持 GPU 加速（NVIDIA GPU 推荐）
- 🔒 支持访问令牌认证
- 📊 自动混合精度计算（BF16/FP16/FP32）
- 🐳 Docker 容器化部署
- 🔍 提供 RESTful API 接口
- 💪 支持多工作进程
- 🩺 健康检查接口

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

### 3. 配置

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
WORKER_CLASS=uvicorn.workers.UvicornWorker
WORKER_CONNECTIONS=1000
TIMEOUT=120
KEEP_ALIVE=5

# 模型配置
MODEL_NAME_OR_PATH=./models/Qwen3-Reranker-4B
MODEL_DEVICE=cuda  # auto, cuda, cpu
MODEL_PRECISION=bfloat16  # 自动检测，支持 bfloat16, float16, float32

# 显存优化配置
MAX_MEMORY_PERCENTAGE=50  # 限制显存使用百分比
BATCH_SIZE=1  # 批处理大小
MAX_SEQ_LENGTH=256  # 最大序列长度

# 性能优化
TOKENIZERS_PARALLELISM=false  # 禁用 tokenizer 并行
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 优化 CUDA 内存分配

# 访问令牌（可选）
# ACCESS_TOKEN=your-secret-token

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### 4. 启动服务

使用 Docker Compose 启动服务：

```bash
docker-compose up -d --build
```

### 5. 验证服务

检查服务健康状态：

```bash
curl http://localhost:6006/health
```

正常响应：
```json
{
  "status": "ok"
}
```

## 📚 API 文档

### 1. 重排序文档

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
  "documents": ["文档1", "文档2", "文档3"],
  "top_k": 3
}
```

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| query | string | 是 | 查询文本 |
| documents | array | 是 | 需要排序的文档列表 |
| top_k | integer | 否 | 返回最相关的 k 个结果，默认返回所有 |

**示例请求**

```bash
curl -X 'POST' \
  'http://localhost:6006/v1/rerank' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "什么是人工智能？",
    "documents": [
      "人工智能是计算机科学的一个分支。",
      "机器学习是人工智能的一个子领域。",
      "深度学习是机器学习的一种方法。"
    ],
    "top_k": 2
  }'
```

**成功响应**

```json
[
  {
    "index": 0,
    "score": 0.8765
  },
  {
    "index": 1,
    "score": 0.7654
  }
]
```

### 2. 健康检查

**端点**

```
GET /health
```

**响应**

```json
{
  "status": "ok"
}
```

## 📜 许可证

本项目采用 [MIT License](LICENSE) 许可证。

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

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
| `MODEL_DEVICE` | `cuda` | 运行设备 (`cuda`/`cpu`) |
| `MODEL_PRECISION` | `bfloat16` | 模型精度 (`bfloat16`/`float16`/`float32`) |
| `MAX_MEMORY_PERCENTAGE` | `50` | 显存使用百分比 |
| `BATCH_SIZE` | `1` | 批处理大小 |
| `MAX_SEQ_LENGTH` | `256` | 最大序列长度 |
| `ACCESS_TOKEN` | `None` | 访问令牌（可选） |

## 📄 许可证

MIT
    "documents": [
      "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统。",
      "机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。",
      "深度学习是机器学习的一个子集，使用神经网络进行学习。"
    ],
    "top_k": 2
  }'
```

**响应示例**

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "text": "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统。"
    },
    {
      "index": 1,
      "relevance_score": 0.88,
      "text": "机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。"
    }
  ]
}
```

### 2. 健康检查

**端点**

```
GET /health
```

**示例请求**

```bash
curl http://localhost:6006/health
```

**响应示例**

```json
{
  "status": "ok"
}
```

## 客户端示例

### Python 示例

```python
import requests

def rerank(query, documents, top_k=None, api_key="your_api_key"):
    url = "http://localhost:6006/api/v1/rerank"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "query": query,
        "documents": documents
    }
    
    if top_k is not None:
        data["top_k"] = top_k
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 使用示例
results = rerank(
    query="什么是人工智能？",
    documents=[
        "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统。",
        "机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。",
        "深度学习是机器学习的一个子集，使用神经网络进行学习。"
    ],
    top_k=2
)
print(results)
```

## 性能优化

1. **调整批处理大小**：
   - 在 `.env` 中修改 `BATCH_SIZE` 参数
   - 显存不足时减小此值

2. **调整工作进程数**：
   - 在 `.env` 中修改 `WORKERS` 参数
   - 通常设置为 GPU 数量

3. **精度设置**：
   - 使用 `bf16` 或 `fp16` 可以减少显存使用
   - 在 `.env` 中修改 `MODEL_PRECISION`

## 常见问题

### 1. 显存不足

- 减小 `BATCH_SIZE`
- 减小 `MAX_MEMORY_PERCENTAGE`
- 使用更小的模型
- 使用 CPU 模式（性能较低）

### 2. 请求超时

- 增加 `TIMEOUT` 参数
- 减少批处理大小

## 许可证

[您的许可证信息]
