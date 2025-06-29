# 使用基础镜像
FROM ccr.ccs.tencentyun.com/waveman/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 设置工作目录
WORKDIR /app

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean



# 复制项目文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --default-timeout=100 \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
    -r requirements.txt

# 创建模型目录
RUN mkdir -p /app/models

# 复制模型文件（使用 .dockerignore 确保只复制需要的文件）
COPY models/ /app/models/

# 复制项目文件
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=6006 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    TIMEOUT=300 \
    KEEP_ALIVE=10 \
    MODEL_DEVICE=cuda \
    MODEL_PRECISION=bf16 \
    MAX_MEMORY_PERCENTAGE=50 \
    BATCH_SIZE=1 \
    MAX_SEQ_LENGTH=256 \
    LOG_LEVEL=INFO \
    LOG_FORMAT='%(asctime)s - %(name)s - %(levelname)s - %(message)s' \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    # 设置模型路径
    MODEL_NAME_OR_PATH=/app/models/Qwen3-Reranker-4B

# 创建日志目录
RUN mkdir -p /app/logs

# 暴露端口
EXPOSE ${PORT}

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# 启动命令
CMD ["python", "app.py"]