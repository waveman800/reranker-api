#!/bin/bash
# 多模态重排服务启动脚本

set -e

echo "=========================================="
echo "多模态重排服务启动脚本"
echo "=========================================="

# 检查环境
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，正在从 .env.example 复制..."
    cp .env.example .env
    echo "✅ 已创建 .env 文件，请根据需要修改配置"
fi

# 检查模型目录
if [ ! -d "models" ]; then
    echo "📁 创建模型目录..."
    mkdir -p models
fi

# 检查模型文件
if [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "⚠️  模型目录为空，请下载模型文件到 models/ 目录"
    echo ""
    echo "下载命令示例："
    echo "  huggingface-cli download Qwen/Qwen3-VL-Reranker-2B --local-dir ./models/Qwen3-VL-Reranker-2B"
    echo ""
    echo "或者："
    echo "  modelscope download --model qwen/Qwen3-VL-Reranker-2B --local_dir ./models/Qwen3-VL-Reranker-2B"
    echo ""
    read -p "是否继续启动？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 检查 NVIDIA Docker 运行时（可选）
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  未检测到 NVIDIA Docker 运行时，GPU 加速可能不可用"
    echo "   如需 GPU 支持，请安装 nvidia-docker2"
fi

# 构建并启动服务
echo ""
echo "🚀 正在构建并启动服务..."
docker-compose up -d --build

echo ""
echo "✅ 服务已启动！"
echo ""
echo "服务地址: http://localhost:6006"
echo ""
echo "常用命令："
echo "  查看日志: docker-compose logs -f"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
echo "  查看状态: docker-compose ps"
echo ""
echo "测试服务："
echo "  curl http://localhost:6006/health"
echo ""

# 等待服务启动
sleep 3

# 健康检查
if curl -s http://localhost:6006/health > /dev/null; then
    echo "✅ 健康检查通过！"
    curl -s http://localhost:6006/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:6006/health
else
    echo "⚠️  服务可能还在启动中，请稍后检查"
fi
