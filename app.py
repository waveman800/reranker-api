import os
import time
import logging
import threading
import base64
import io
import numpy as np
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS, HTTP_400_BAD_REQUEST
from contextlib import asynccontextmanager
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor

# 应用配置
class RerankerConfig:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME_OR_PATH", "/app/models/Qwen3-VL-Reranker-2B")
        self.device = os.getenv("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.precision = os.getenv("MODEL_PRECISION", "").lower()  # 默认为空，表示自动检测
        self.max_length = int(os.getenv("MAX_SEQ_LENGTH", 512))
        self.batch_size = int(os.getenv("BATCH_SIZE", 4))
        self.access_token = os.getenv("ACCESS_TOKEN", "")
        self.max_memory_percent = float(os.getenv("MAX_MEMORY_PERCENTAGE", 80))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.model_type = os.getenv("MODEL_TYPE", "multimodal").lower()  # text 或 multimodal

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
    title="Multimodal Reranker API",
    description="支持文本和多模态的重排序服务",
    version="2.0.0"
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
class Document(BaseModel):
    """文档模型，支持文本和图像"""
    text: Optional[str] = Field(None, description="文档文本内容")
    image_url: Optional[str] = Field(None, description="图像URL或base64编码的图像")
    image_type: Optional[str] = Field("url", description="图像类型: url 或 base64")

    @validator('text', 'image_url')
    def check_content(cls, v, values):
        # 确保至少有一个字段有值
        return v

class RerankRequest(BaseModel):
    """重排序请求模型"""
    query: str = Field(..., description="查询文本")
    documents: List[Document] = Field(..., description="文档列表，每个文档可以是文本或图像")
    top_k: Optional[int] = Field(None, description="返回最相关的k个结果")
    return_documents: Optional[bool] = Field(False, description="是否返回原始文档内容")

class RerankResult(BaseModel):
    """重排序结果模型"""
    index: int = Field(..., description="文档原始索引")
    score: float = Field(..., description="相关性分数")
    document: Optional[Document] = Field(None, description="原始文档内容（如果请求中设置了return_documents=True）")

class RerankResponse(BaseModel):
    """重排序响应模型"""
    results: List[RerankResult]
    model_type: str = Field(..., description="使用的模型类型")

# 图像处理工具类
class ImageProcessor:
    """图像处理工具类"""
    
    @staticmethod
    def load_image(image_source: str, image_type: str = "url") -> Image.Image:
        """加载图像，支持URL和base64编码"""
        try:
            if image_type == "base64":
                # 处理base64编码的图像
                if image_source.startswith("data:image"):
                    # 移除data URI scheme前缀
                    image_source = image_source.split(",")[1]
                image_data = base64.b64decode(image_source)
                image = Image.open(io.BytesIO(image_data))
            else:
                # 假设是文件路径或URL
                if os.path.exists(image_source):
                    image = Image.open(image_source)
                else:
                    # 尝试作为base64处理（容错）
                    try:
                        image_data = base64.b64decode(image_source)
                        image = Image.open(io.BytesIO(image_data))
                    except:
                        raise ValueError(f"无法加载图像: {image_source}")
            
            # 转换为RGB模式（处理RGBA、P等模式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"图像加载失败: {e}")
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=f"图像加载失败: {str(e)}")

# 模型加载器基类
class BaseReranker:
    """重排模型基类"""
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None, return_documents: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

# 纯文本重排模型
class TextReranker(BaseReranker):
    """基于Qwen3-Reranker的纯文本重排模型"""
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
        logger.info(f"Loading text reranker model from {config.model_name}...")
        start_time = time.time()
        
        # 设置设备
        self.device = torch.device(config.device)
        
        # 加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True
        )
        
        # 尝试从模型配置中获取torch_dtype
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"BF16 supported: {hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported()}")
            
            # 从配置中获取torch_dtype
            if hasattr(model_config, "torch_dtype"):
                if isinstance(model_config.torch_dtype, str):
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
                    torch_dtype = model_config.torch_dtype
                    logger.info(f"Using model's default dtype: {torch_dtype}")
            else:
                torch_dtype = {
                    "bf16": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                    "fp16": torch.float16,
                    "fp32": torch.float32
                }.get(config.precision, None)
                
                if torch_dtype is None:
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
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: f"{max_memory}MB"}
        
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded with dtype: {next(self.model.parameters()).dtype}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        self.model.eval()
        
        logger.info(f"Text reranker model loaded in {time.time() - start_time:.2f} seconds")
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None, return_documents: bool = False) -> Dict[str, Any]:
        """对文本文档进行重排序"""
        if not documents:
            return {"results": [], "model_type": "text"}
        
        # 提取文本内容
        texts = []
        for doc in documents:
            if doc.text:
                texts.append(doc.text)
            elif doc.image_url:
                # 对于纯文本模型，如果文档是图像，使用占位符
                texts.append("[图像内容]")
            else:
                texts.append("")
        
        # 准备输入
        pairs = [[query, text] for text in texts]
        
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
                
                logits = outputs.logits
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                scores = logits.float().cpu().numpy()
                
                if isinstance(scores, np.ndarray) and scores.size > 1:
                    scores = scores.ravel()
                elif isinstance(scores, (list, tuple)):
                    scores = np.array(scores).ravel()
                elif not isinstance(scores, np.ndarray):
                    scores = np.array([scores])
            
            # 处理结果
            for j, score in enumerate(scores):
                result = {"index": i + j, "score": float(score)}
                if return_documents:
                    result["document"] = documents[i + j]
                results.append(result)
        
        # 归一化处理
        if results:
            scores = np.array([r["score"] for r in results])
            min_score = np.min(scores)
            max_score = np.max(scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                for r in results:
                    r["score"] = (r["score"] - min_score) / score_range
            else:
                for r in results:
                    r["score"] = 0.5
        
        # 排序并限制top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return {"results": results, "model_type": "text"}

# 多模态重排模型
class MultimodalReranker(BaseReranker):
    """基于Qwen3-VL-Reranker的多模态重排模型"""
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
        logger.info(f"Loading multimodal reranker model from {config.model_name}...")
        start_time = time.time()
        
        # 设置设备
        self.device = torch.device(config.device)
        
        # 加载processor和model
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 尝试从模型配置中获取torch_dtype
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"BF16 supported: {hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported()}")
            
            # 从配置中获取torch_dtype
            if hasattr(model_config, "torch_dtype"):
                if isinstance(model_config.torch_dtype, str):
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
                    torch_dtype = model_config.torch_dtype
                    logger.info(f"Using model's default dtype: {torch_dtype}")
            else:
                torch_dtype = {
                    "bf16": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                    "fp16": torch.float16,
                    "fp32": torch.float32
                }.get(config.precision, None)
                
                if torch_dtype is None:
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
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: f"{max_memory}MB"}
        
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        
        # 尝试加载多模态模型
        try:
            # 首先尝试Qwen2.5-VL模型
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_name,
                **model_kwargs
            )
            logger.info("Loaded model as Qwen2_5_VLForConditionalGeneration")
        except Exception as e:
            logger.warning(f"Failed to load as Qwen2_5_VLForConditionalGeneration: {e}")
            try:
                # 尝试AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    config.model_name,
                    **model_kwargs
                )
                logger.info("Loaded model as AutoModelForVision2Seq")
            except Exception as e2:
                logger.warning(f"Failed to load as AutoModelForVision2Seq: {e2}")
                # 最后尝试AutoModelForSequenceClassification
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    config.model_name,
                    **model_kwargs
                )
                logger.info("Loaded model as AutoModelForSequenceClassification")
        
        self.model = self.model.to(self.device)
        
        logger.info(f"Model loaded with dtype: {next(self.model.parameters()).dtype}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        self.model.eval()
        
        self.image_processor = ImageProcessor()
        
        logger.info(f"Multimodal reranker model loaded in {time.time() - start_time:.2f} seconds")
    
    def _prepare_multimodal_input(self, query: str, document: Document) -> Dict[str, Any]:
        """准备多模态输入"""
        if document.image_url:
            # 加载图像
            image = self.image_processor.load_image(document.image_url, document.image_type or "url")
            
            # 构建多模态prompt
            text_content = document.text or ""
            if text_content:
                prompt = f"Query: {query}\nDocument Text: {text_content}\nRelevance score:"
            else:
                prompt = f"Query: {query}\nImage Content\nRelevance score:"
            
            return {
                "text": prompt,
                "images": [image]
            }
        else:
            # 纯文本输入
            prompt = f"Query: {query}\nDocument: {document.text}\nRelevance score:"
            return {
                "text": prompt,
                "images": None
            }
    
    def rerank(self, query: str, documents: List[Document], top_k: Optional[int] = None, return_documents: bool = False) -> Dict[str, Any]:
        """对多模态文档进行重排序"""
        if not documents:
            return {"results": [], "model_type": "multimodal"}
        
        # 获取模型当前精度
        dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device
        
        # 批处理
        results = []
        for i in range(0, len(documents), config.batch_size):
            batch_docs = documents[i:i + config.batch_size]
            batch_scores = []
            
            for doc in batch_docs:
                try:
                    # 准备输入
                    input_data = self._prepare_multimodal_input(query, doc)
                    
                    # 使用processor处理输入
                    if input_data["images"]:
                        inputs = self.processor(
                            text=[input_data["text"]],
                            images=input_data["images"],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=config.max_length
                        ).to(device)
                    else:
                        inputs = self.processor(
                            text=[input_data["text"]],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=config.max_length
                        ).to(device)
                    
                    # 确保输入张量使用正确的数据类型
                    for key in inputs.keys():
                        if inputs[key].dtype == torch.float32 or inputs[key].dtype == torch.float16:
                            inputs[key] = inputs[key].to(dtype)
                        inputs[key] = inputs[key].to(device)
                    
                    # 推理
                    with torch.cuda.amp.autocast(dtype=dtype):
                        with torch.no_grad():
                            # 根据模型类型处理输出
                            if hasattr(self.model, 'score'):
                                # 如果是专门的reranker模型
                                outputs = self.model(**inputs)
                                if hasattr(outputs, 'logits'):
                                    score = outputs.logits[0].item()
                                else:
                                    score = outputs[0].item()
                            else:
                                # 对于生成式模型，使用特定方式获取分数
                                outputs = self.model.generate(
                                    **inputs,
                                    max_new_tokens=10,
                                    return_dict_in_generate=True,
                                    output_scores=True
                                )
                                # 从生成结果中提取分数
                                score = self._extract_score_from_generation(outputs)
                    
                    batch_scores.append(score)
                    
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
                    batch_scores.append(0.0)
            
            # 处理结果
            for j, score in enumerate(batch_scores):
                result = {"index": i + j, "score": float(score)}
                if return_documents:
                    result["document"] = documents[i + j]
                results.append(result)
        
        # 归一化处理
        if results:
            scores = np.array([r["score"] for r in results])
            min_score = np.min(scores)
            max_score = np.max(scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                for r in results:
                    r["score"] = (r["score"] - min_score) / score_range
            else:
                for r in results:
                    r["score"] = 0.5
        
        # 排序并限制top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return {"results": results, "model_type": "multimodal"}
    
    def _extract_score_from_generation(self, outputs) -> float:
        """从生成结果中提取分数"""
        try:
            # 尝试从scores中提取
            if hasattr(outputs, 'scores') and outputs.scores:
                # 获取第一个token的分数分布
                scores = outputs.scores[0][0]
                # 使用softmax归一化
                probs = torch.softmax(scores, dim=-1)
                # 返回最高概率作为相关性分数
                return probs.max().item()
            return 0.5
        except Exception as e:
            logger.warning(f"Failed to extract score from generation: {e}")
            return 0.5

# 模型工厂
class RerankerFactory:
    """重排模型工厂"""
    
    @staticmethod
    def create_reranker(model_type: str = None) -> BaseReranker:
        """创建重排模型实例"""
        model_type = model_type or config.model_type
        
        if model_type == "multimodal":
            return MultimodalReranker()
        elif model_type == "text":
            return TextReranker()
        else:
            # 自动检测
            try:
                return MultimodalReranker()
            except Exception as e:
                logger.warning(f"Failed to load multimodal model: {e}, falling back to text model")
                return TextReranker()

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    app.state.reranker = RerankerFactory.create_reranker()
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

# API端点 - JSON格式
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_json(
    request: RerankRequest,
    background_tasks: BackgroundTasks,
    request_obj: Request = None,
    token: Any = Depends(verify_token) if config.access_token else None
):
    """重排序API端点（JSON格式）"""
    try:
        reranker = app.state.reranker
        results = reranker.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
            return_documents=request.return_documents
        )
        return RerankResponse(**results)
    except Exception as e:
        logger.error(f"Rerank error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API端点 - 兼容旧版纯文本格式
@app.post("/v1/rerank/text")
async def rerank_text(
    query: str = Form(..., description="查询文本"),
    documents: List[str] = Form(..., description="文档列表"),
    top_k: Optional[int] = Form(None, description="返回最相关的k个结果"),
    token: Any = Depends(verify_token) if config.access_token else None
):
    """重排序API端点（兼容旧版纯文本格式）"""
    try:
        reranker = app.state.reranker
        # 将字符串列表转换为Document列表
        doc_list = [Document(text=doc) for doc in documents]
        results = reranker.rerank(
            query=query,
            documents=doc_list,
            top_k=top_k,
            return_documents=False
        )
        return results
    except Exception as e:
        logger.error(f"Rerank error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API端点 - 支持文件上传的多模态重排序
@app.post("/v1/rerank/multimodal")
async def rerank_multimodal(
    query: str = Form(..., description="查询文本"),
    top_k: Optional[int] = Form(None, description="返回最相关的k个结果"),
    return_documents: bool = Form(False, description="是否返回原始文档"),
    documents_text: Optional[List[str]] = Form(None, description="文档文本列表"),
    images: Optional[List[UploadFile]] = File(None, description="图像文件列表"),
    token: Any = Depends(verify_token) if config.access_token else None
):
    """重排序API端点（支持文件上传的多模态格式）"""
    try:
        reranker = app.state.reranker
        doc_list = []
        
        # 处理文本文档
        if documents_text:
            for text in documents_text:
                doc_list.append(Document(text=text))
        
        # 处理图像文件
        if images:
            for image in images:
                # 读取图像文件并转换为base64
                content = await image.read()
                image_base64 = base64.b64encode(content).decode('utf-8')
                doc_list.append(Document(
                    image_url=f"data:{image.content_type};base64,{image_base64}",
                    image_type="base64"
                ))
        
        if not doc_list:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="No documents provided")
        
        results = reranker.rerank(
            query=query,
            documents=doc_list,
            top_k=top_k,
            return_documents=return_documents
        )
        return RerankResponse(**results)
    except Exception as e:
        logger.error(f"Rerank error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_type": config.model_type,
        "device": config.device
    }

# 模型信息端点
@app.get("/v1/model/info")
async def model_info():
    """获取模型信息"""
    return {
        "model_name": config.model_name,
        "model_type": config.model_type,
        "device": config.device,
        "precision": config.precision,
        "max_length": config.max_length,
        "batch_size": config.batch_size
    }

# 主函数
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 6006)),
        workers=int(os.getenv("WORKERS", 1))
    )
