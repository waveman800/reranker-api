#!/usr/bin/env python3
"""
模型下载脚本

支持从 Hugging Face 或 ModelScope 下载模型

使用方法:
    python download_model.py --model Qwen/Qwen3-VL-Reranker-2B --source huggingface
    python download_model.py --model qwen/Qwen3-VL-Reranker-2B --source modelscope
"""

import argparse
import os
import sys
from pathlib import Path


def download_from_huggingface(model_name: str, local_dir: str):
    """从 Hugging Face 下载模型"""
    try:
        from huggingface_hub import snapshot_download
        print(f"📥 正在从 Hugging Face 下载模型: {model_name}")
        print(f"📁 保存到: {local_dir}")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ 模型下载完成: {local_dir}")
        return True
    except ImportError:
        print("❌ 请先安装 huggingface_hub: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def download_from_modelscope(model_name: str, local_dir: str):
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        print(f"📥 正在从 ModelScope 下载模型: {model_name}")
        print(f"📁 保存到: {local_dir}")
        
        snapshot_download(
            model_name,
            cache_dir=local_dir
        )
        print(f"✅ 模型下载完成: {local_dir}")
        return True
    except ImportError:
        print("❌ 请先安装 modelscope: pip install modelscope")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载多模态重排模型")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-Reranker-2B",
        help="模型名称 (默认: Qwen/Qwen3-VL-Reranker-2B)"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="下载源 (默认: huggingface)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="输出目录 (默认: ./models)"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取模型短名称
    model_short_name = args.model.split("/")[-1]
    local_dir = output_dir / model_short_name
    
    print("=" * 60)
    print("多模态重排模型下载工具")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"来源: {args.source}")
    print(f"输出: {local_dir}")
    print("=" * 60)
    
    # 检查是否已存在
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"⚠️  模型目录已存在: {local_dir}")
        response = input("是否重新下载？(y/N): ")
        if response.lower() != 'y':
            print("已取消下载")
            return
    
    # 下载模型
    if args.source == "huggingface":
        success = download_from_huggingface(args.model, str(local_dir))
    else:
        success = download_from_modelscope(args.model, str(local_dir))
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 模型下载成功！")
        print(f"模型路径: {local_dir}")
        print("\n更新 .env 文件中的模型配置:")
        print(f'  MODEL_NAME_OR_PATH=./models/{model_short_name}')
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 模型下载失败")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
