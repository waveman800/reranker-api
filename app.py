import os
import time
import logging
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS
from contextlib import asynccontextmanager
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor

# 应用配置
class RerankerConfig:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME_OR_PATH", "/app/models/Qwen3-Reranker-4B")
        self.device = os.getenv("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.precision = os.getenv("MODEL_PRECISION", "").lower()  # 默认为空，表示自动检测
        self.max_length = int(os.getenv("MAX_SEQ_LENGTH", 512))
        self.batch_size = int(os.getenv("BATCH_SIZE", 16))
        self.access_token = os.getenv("ACCESS_TOKEN", "")
        self.max_memory_percent = float(os.getenv("MAX_MEMORY_PERCENTAGE", 80))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

# 初始化配置
config = RerankerConfig()

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(
    title="Reranker API",
    description="高性能的重排序服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求和响应模型
class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None

class RerankResult(BaseModel):
    index: int
    score: float

# Removed RerankResponse as we're now returning List[RerankResult] directly

# 模型加载器
class Reranker:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        logger.info(f"Loading model from {config.model_name}...")
        start_time = time.time()
        
        # 设置设备
        self.device = torch.device(config.device)
        
        # 加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            use_fast=True,  # 使用更快的tokenizer实现
            local_files_only=True  # 强制只使用本地文件
        )
        
        # 尝试从模型配置中获取torch_dtype
        try:
            # 先加载配置
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            
            # 记录当前环境支持的精度
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"BF16 supported: {hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported()}")
            
            # 从配置中获取torch_dtype
            if hasattr(model_config, "torch_dtype"):
                if isinstance(model_config.torch_dtype, str):
                    # 处理字符串形式的torch_dtype
                    dtype_str = model_config.torch_dtype.lower()
                    if dtype_str in ["bfloat16", "bf16"]:
                        if hasattr(torch, 'bfloat16') and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                            torch_dtype = torch.bfloat16
                            logger.info("Using bfloat16 precision as specified in model config")
                        else:
                            torch_dtype = torch.float32
                            logger.warning("bfloat16 is not supported on this device, falling back to float32")
                    elif dtype_str in ["float16", "fp16"]:
                        torch_dtype = torch.float16
                    elif dtype_str in ["float32", "fp32"]:
                        torch_dtype = torch.float32
                    else:
                        logger.warning(f"Unsupported torch_dtype in config: {model_config.torch_dtype}, using float32")
                        torch_dtype = torch.float32
                else:
                    # 直接使用torch_dtype（如果是torch.dtype类型）
                    torch_dtype = model_config.torch_dtype
                logger.info(f"Using model's default dtype: {torch_dtype}")
            else:
                # 如果配置中没有指定，则使用环境变量中的设置
                torch_dtype = {
                    "bf16": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                    "fp16": torch.float16,
                    "fp32": torch.float32
                }.get(config.precision, None)
                
                if torch_dtype is None:
                    # 如果环境变量中也没有指定，则根据CUDA可用性自动选择
                    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                logger.info(f"Using dtype from config: {torch_dtype}")
                
        except Exception as e:
            logger.warning(f"Failed to get dtype from model config: {e}, using float32 as fallback")
            torch_dtype = torch.float32
        
        # 设置显存限制
        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = int(total_memory * (config.max_memory_percent / 100))
            logger.info(f"Limiting GPU memory usage to {max_memory/1024**3:.2f}GB")
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,  # 信任远程代码
            "low_cpu_mem_usage": True,  # 减少CPU内存使用
        }
        
        # 设置设备
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: f"{max_memory}MB"}
        
        # 加载模型时指定torch_dtype
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        
        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        logger.info(f"Model loaded with dtype: {next(self.model.parameters()).dtype}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        
        # 启用梯度检查点以节省显存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        self.model.eval()
        
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        # 准备输入
        pairs = [[query, doc] for doc in documents]
        
        # 获取模型当前精度
        dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device
        
        # 确保模型在正确的设备上
        self.model = self.model.to(device)
        
        # 批处理
        results = []
        for i in range(0, len(pairs), config.batch_size):
            batch_pairs = pairs[i:i + config.batch_size]
            
            # 编码
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt"
            ).to(device)
            
            # 确保输入张量使用正确的数据类型
            if hasattr(inputs, 'input_ids'):
                inputs.input_ids = inputs.input_ids.to(device)
            if hasattr(inputs, 'attention_mask'):
                inputs.attention_mask = inputs.attention_mask.to(device)
            if hasattr(inputs, 'token_type_ids'):
                inputs.token_type_ids = inputs.token_type_ids.to(device)
            
            # 使用自动混合精度
            with torch.cuda.amp.autocast(dtype=dtype):
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # 处理模型输出，确保是一维数组
                logits = outputs.logits
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                scores = logits.float().cpu().numpy()
                
                # 确保scores是一维数组
                if isinstance(scores, np.ndarray) and scores.size > 1:
                    scores = scores.ravel()
                elif isinstance(scores, (list, tuple)):
                    scores = np.array(scores).ravel()
                elif not isinstance(scores, np.ndarray):
                    scores = np.array([scores])
            
            # 处理结果
            for j, score in enumerate(scores):
                results.append({"index": i + j, "score": float(score)})
        
        # 如果结果不为空，进行归一化处理
        if results:
            # 提取所有分数
            scores = np.array([r["score"] for r in results])
            
            # 计算最小值和范围
            min_score = np.min(scores)
            max_score = np.max(scores)
            score_range = max_score - min_score
            
            # 避免除以零（当所有分数相同时）
            if score_range > 0:
                # 对每个结果进行归一化
                for r in results:
                    r["score"] = (r["score"] - min_score) / score_range
            else:
                # 如果所有分数相同，则都设为0.5
                for r in results:
                    r["score"] = 0.5
        
        # 如果指定了top_k，只返回前k个结果
        if top_k is not None and top_k > 0:
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
            
        return {"results": results}

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    app.state.reranker = Reranker()
    yield
    # 清理资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app.router.lifespan_context = lifespan

# 验证访问令牌
async def verify_token(request: Request):
    if not config.access_token:
        return None
        
    # 从查询参数获取 token
    token = request.query_params.get("access_token")
    
    # 如果查询参数中没有，尝试从 Authorization 头获取
    if not token and "authorization" in request.headers:
        auth_header = request.headers["authorization"]
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    if not token or token != config.access_token:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing access token"
        )

# API端点
@app.post("/v1/rerank", response_model=List[RerankResult])
async def rerank(
    request: RerankRequest,
    background_tasks: BackgroundTasks,
    request_obj: Request = None,
    token: Any = Depends(verify_token) if config.access_token else None
):
    try:
        reranker = app.state.reranker
        results = reranker.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )
        # Remove the 'results' wrapper and return the list directly
        if isinstance(results, dict) and 'results' in results:
            return results['results']
        return results
    except Exception as e:
        logger.error(f"Rerank error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 主函数
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 6006)),
        workers=int(os.getenv("WORKERS", 1))
    )