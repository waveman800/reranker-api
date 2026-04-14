import os
import json
import logging
from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional, Literal
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=6006)
    workers: int = Field(
        default=1,  # 对于 GPU 服务，建议设置为 1 以避免 GPU 内存竞争
    )
    worker_connections: int = Field(
        default=100,  # 减少并发连接数以节省内存
    )
    timeout: int = Field(
        default=300,  # 增加超时时间以支持长时间运行的推理任务
    )
    keep_alive: int = Field(
        default=10,  # 适当增加 keep-alive 时间
    )
    max_requests: int = Field(
        default=1000,  # 处理 1000 个请求后重启 worker 防止内存泄漏
    )
    max_requests_jitter: int = Field(
        default=100,  # 添加随机抖动，避免所有 worker 同时重启
    )
    reload: bool = Field(
        default=False,
    )
    access_token: str = Field(
        default="",
    )

    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "6006")),
            workers=int(os.getenv("WORKERS", "1")),
            worker_connections=int(os.getenv("WORKER_CONNECTIONS", "100")),
            timeout=int(os.getenv("TIMEOUT", "300")),
            keep_alive=int(os.getenv("KEEP_ALIVE", "10")),
            max_requests=int(os.getenv("MAX_REQUESTS", "1000")),
            max_requests_jitter=int(os.getenv("MAX_REQUESTS_JITTER", "100")),
            reload=os.getenv("RELOAD", "false").lower() == "true",
            access_token=os.getenv("ACCESS_TOKEN", ""),
        )

class ModelConfig(BaseModel):
    name_or_path: str = Field(default="./models/Qwen3-VL-Reranker-2B")
    device: str = Field(default="cuda")
    precision: Optional[Literal["bf16", "bfloat16", "fp16", "fp32"]] = Field(default=None)
    max_seq_length: int = Field(default=512)
    batch_size: int = Field(default=4)
    torch_dtype: Optional[str] = Field(default=None)
    model_type: Literal["text", "multimodal", "auto"] = Field(default="multimodal")
    
    @field_validator('torch_dtype', mode='before')
    @classmethod
    def set_torch_dtype(cls, v, info):
        """设置 torch_dtype 基于 precision"""
        values = info.data
        precision = values.get('precision')
        if precision == 'bfloat16':
            return 'bfloat16'
        elif precision == 'fp16':
            return 'float16'
        elif precision == 'fp32':
            return 'float32'
        return 'bfloat16' if values.get('device') == 'cuda' else 'float32'
    
    @field_validator('precision', mode='before')
    @classmethod
    def set_precision(cls, v):
        """确保 precision 使用标准格式"""
        if v == 'bfloat16':
            return 'bf16'
        return v or 'bf16'

    @classmethod
    def from_env(cls):
        return cls(
            name_or_path=os.getenv("MODEL_NAME_OR_PATH", "./models/Qwen3-VL-Reranker-2B"),
            device=os.getenv("MODEL_DEVICE", "cuda"),
            precision=os.getenv("MODEL_PRECISION", None),
            max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512")),
            batch_size=int(os.getenv("BATCH_SIZE", "4")),
            model_type=os.getenv("MODEL_TYPE", "multimodal"),
        )

class LogConfig(BaseModel):
    level: str = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    @classmethod
    def from_env(cls):
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        )

class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    log: LogConfig = LogConfig()

    @classmethod
    def from_env(cls):
        return cls(
            server=ServerConfig.from_env(),
            model=ModelConfig.from_env(),
            log=LogConfig.from_env(),
        )

# 全局配置实例
config = Config.from_env()

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.log.level),
    format=config.log.format
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 打印当前配置
    print(json.dumps(config.dict(), indent=2))
