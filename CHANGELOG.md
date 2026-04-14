# 变更日志 (Changelog)

## v2.0.0 - 多模态重排支持

### ✨ 新增功能

#### 1. 多模态重排支持
- **新增 `MultimodalReranker` 类**: 支持 Qwen3-VL-Reranker-2B 多模态模型
- **图像处理**: 支持图像URL、Base64编码图像、文件上传
- **图文混合**: 支持同时包含文本和图像的文档

#### 2. 新的 API 端点
- **`POST /v1/rerank`**: 多模态重排序（JSON格式）
  - 支持文本文档
  - 支持图像URL
  - 支持Base64编码图像
  - 支持图文混合文档
  
- **`POST /v1/rerank/text`**: 纯文本重排序（向后兼容）
  - 兼容旧版API
  - 使用 Form 格式
  
- **`POST /v1/rerank/multimodal`**: 文件上传重排序
  - 支持上传图像文件
  - 支持混合文本文档和图像文件

- **`GET /v1/model/info`**: 模型信息查询
  - 获取当前模型配置

#### 3. 模型类型支持
- **`TextReranker`**: 纯文本重排（Qwen3-Reranker-4B）
- **`MultimodalReranker`**: 多模态重排（Qwen3-VL-Reranker-2B）
- **`RerankerFactory`**: 自动检测和创建模型实例

#### 4. 配置文件更新
- 新增 `MODEL_TYPE` 环境变量（text/multimodal/auto）
- 默认模型改为 `Qwen3-VL-Reranker-2B`
- 更新默认批处理大小为 4（多模态优化）

#### 5. 新增工具脚本
- **`download_model.py`**: 模型下载工具
  - 支持 Hugging Face 和 ModelScope
  - 自动创建模型目录
  
- **`client_example.py`**: 客户端示例
  - 完整的 Python 客户端类
  - 多种使用示例
  
- **`test_api.py`**: API 测试脚本
  - 全面的 API 测试
  - 支持 pytest
  
- **`start.sh`**: 一键启动脚本
  - 自动检查环境
  - 自动构建和启动

#### 6. Docker 优化
- 更新 Dockerfile 支持多模态依赖
- 添加图像处理库（libgl1-mesa-glx, libglib2.0-0）
- 优化 .dockerignore

### 🔧 修改内容

#### app.py
- 完全重构，支持多模态输入
- 新增 `Document` 数据模型
- 新增 `ImageProcessor` 图像处理工具
- 新增 `BaseReranker` 抽象基类
- 新增 `TextReranker` 和 `MultimodalReranker` 实现类
- 新增 `RerankerFactory` 工厂类
- 更新 API 端点支持多种格式

#### config.py
- 更新为 Pydantic v2 语法
- 新增 `model_type` 配置项
- 优化配置验证器

#### requirements.txt
- 更新 transformers 版本要求
- 新增 Pillow 图像处理库
- 新增 qwen-vl-utils 工具库
- 更新 Pydantic 到 v2

#### docker-compose.yml
- 更新默认模型路径
- 新增 `MODEL_TYPE` 环境变量
- 优化批处理大小

#### Dockerfile
- 添加图像处理系统依赖
- 更新默认模型路径
- 新增 `MODEL_TYPE` 环境变量

#### README.md
- 完全重写，包含多模态使用说明
- 新增 API 文档
- 新增客户端示例
- 新增常见问题

### 🔄 向后兼容

- 保留 `/v1/rerank/text` 端点，兼容旧版纯文本 API
- 支持通过 `MODEL_TYPE=text` 使用纯文本模式
- 原有配置文件格式仍然有效

### 📋 升级指南

1. **更新代码**
   ```bash
   git pull
   ```

2. **更新依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **下载多模态模型**（可选）
   ```bash
   python download_model.py --model Qwen/Qwen3-VL-Reranker-2B
   ```

4. **更新配置**
   ```bash
   # 编辑 .env 文件
   MODEL_NAME_OR_PATH=./models/Qwen3-VL-Reranker-2B
   MODEL_TYPE=multimodal
   ```

5. **重启服务**
   ```bash
   docker-compose up -d --build
   ```

### 📝 注意事项

1. **显存要求**: 多模态模型需要更多显存，建议至少 16GB
2. **批处理大小**: 多模态建议设置较小的 batch_size（如4）
3. **图像格式**: 支持 JPG, PNG, WebP, GIF, BMP 等常见格式
4. **模型兼容性**: Qwen3-VL-Reranker-2B 需要 transformers >= 4.45.0

### 🐛 已知问题

- 首次加载多模态模型可能较慢
- 大图像文件可能消耗较多内存
- 某些图像格式可能需要额外转换

### 🔮 未来计划

- [ ] 支持视频输入
- [ ] 支持批量图像处理
- [ ] 添加更多多模态模型支持
- [ ] 优化图像预处理性能
- [ ] 添加模型热切换功能
