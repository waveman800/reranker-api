import os
import json
import logging
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=6006, env="PORT")
    workers: int = Field(
        default=1,  # 对于 GPU 服务，建议设置为 1 以避免 GPU 内存竞争
        env="WORKERS"
    )
    #worker_class: str = Field(
    #    default="uvicorn.workers.UvicornH11Worker",  # 更轻量级的 worker
    #    env="WORKER_CLASS"
    #)
    worker_connections: int = Field(
        default=100,  # 减少并发连接数以节省内存
        env="WORKER_CONNECTIONS"
    )
    timeout: int = Field(
        default=300,  # 增加超时时间以支持长时间运行的推理任务
        env="TIMEOUT"
    )
    keep_alive: int = Field(
        default=10,  # 适当增加 keep-alive 时间
        env="KEEP_ALIVE"
    )
    max_requests: int = Field(
        default=1000,  # 处理 1000 个请求后重启 worker 防止内存泄漏
        env="MAX_REQUESTS"
    )
    max_requests_jitter: int = Field(
        default=100,  # 添加随机抖动，避免所有 worker 同时重启
        env="MAX_REQUESTS_JITTER"
    )
    reload: bool = Field(
        default=False,
        env="RELOAD"
    )
    access_token: str = Field(
        default="",
        env="ACCESS_TOKEN"
    )

class ModelConfig(BaseModel):
    name_or_path: str = Field(default="./models/Qwen3-Reranker-4B", env="MODEL_NAME_OR_PATH")
    device: str = Field(default="cuda", env="MODEL_DEVICE")
    precision: Optional[Literal["bf16", "bfloat16", "fp16", "fp32"]] = Field(default=None, env="MODEL_PRECISION")
    max_seq_length: int = Field(default=512, env="MAX_SEQ_LENGTH")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    torch_dtype: Optional[str] = Field(default=None)
    
    @validator('torch_dtype', pre=True, always=True)
    def set_torch_dtype(cls, v, values):
        """设置 torch_dtype 基于 precision"""
        precision = values.get('precision')
        if precision == 'bfloat16':
            return 'bfloat16'
        elif precision == 'fp16':
            return 'float16'
        elif precision == 'fp32':
            return 'float32'
        return 'bfloat16' if values.get('device') == 'cuda' else 'float32'
    
    @validator('precision', pre=True, always=True)
    def set_precision(cls, v):
        """确保 precision 使用标准格式"""
        if v == 'bfloat16':
            return 'bf16'
        return v or 'bf16'

class LogConfig(BaseModel):
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )

class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    log: LogConfig = LogConfig()

    @classmethod
    def from_env(cls) -> 'Config':
        """从环境变量加载配置"""
        return cls(
            server=ServerConfig(),
            model=ModelConfig(),
            log=LogConfig()
        )

# 全局配置实例
config = Config.from_env()

def get_config() -> Config:
    """获取配置实例"""
    return config
