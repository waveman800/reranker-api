# Git 提交说明

## 提交概要

```
feat: 重构为支持多模态的重排服务

- 新增 MultimodalReranker 支持 Qwen3-VL-Reranker-2B
- 支持图像URL、Base64编码、文件上传
- 向后兼容纯文本重排序 API
- 新增完整的测试和文档
```

## 详细变更

### 核心功能
- ✨ 新增多模态重排序支持（文本+图像）
- ✨ 新增 MultimodalReranker 类
- ✨ 新增 ImageProcessor 图像处理工具
- ✨ 新增 RerankerFactory 模型工厂
- ✨ 新增 /v1/rerank/multimodal 文件上传端点
- ✨ 新增 /v1/model/info 模型信息端点
- 🔒 向后兼容 /v1/rerank/text 纯文本端点

### API 变更
- POST /v1/rerank - 支持多模态输入（Document模型）
- POST /v1/rerank/text - 纯文本重排序（兼容旧版）
- POST /v1/rerank/multimodal - 文件上传重排序
- GET /v1/model/info - 新增模型信息查询

### 代码重构
- 重构 app.py - 支持多模态架构
- 重构 config.py - 升级 Pydantic v2
- 更新 requirements.txt - 新增多模态依赖
- 更新 docker-compose.yml - 新环境变量
- 更新 Dockerfile - 图像处理依赖

### 新增文件
- client_example.py - Python客户端示例
- test_api.py - API测试脚本
- download_model.py - 模型下载工具
- start.sh - 一键启动脚本
- CHANGELOG.md - 变更日志

### 配置变更
- 新增 MODEL_TYPE 环境变量（text/multimodal/auto）
- 默认模型改为 Qwen3-VL-Reranker-2B
- 优化 BATCH_SIZE 默认为 4（多模态）

### 文档更新
- 完全重写 README.md
- 新增详细 API 文档
- 新增客户端使用示例
- 新增常见问题解答

## 测试状态

- ✅ Python 语法检查通过
- ✅ 代码结构审查通过
- ⏳ 功能测试（待测试专家完成）
- ⏳ 集成测试（待测试专家完成）

## 兼容性

- ✅ 向后兼容纯文本 API
- ✅ 支持 MODEL_TYPE=text 模式
- ✅ 原有配置文件格式有效

## 破坏性变更

无 - 所有变更均为向后兼容的新增功能

## 升级指南

1. 更新代码：`git pull`
2. 更新依赖：`pip install -r requirements.txt`
3. 下载多模态模型（可选）：`python download_model.py`
4. 更新配置：编辑 .env 文件
5. 重启服务：`docker-compose up -d --build`

## 提交命令

```bash
git add -A
git commit -F COMMIT_MESSAGE.md
git push origin main
```
