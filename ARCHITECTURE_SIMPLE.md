# 可信数据护盾 - 简化架构方案

**核心定位**: 数据保险箱 + OpenAI兼容接口，**无需重排功能**

---

## 1. 核心价值

```
┌─────────────────────────────────────────────────────────────────┐
│                     可信数据护盾 = 数据保险箱                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   数据提供方                          大模型厂商/开发者          │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐            │
│   │ 原始数据 │───▶│   TEE保险箱   │───▶│ 模型调用 │            │
│   │ (加密)   │    │  (可用不可见) │    │ (OpenAI) │            │
│   └──────────┘    └──────────────┘    └──────────┘            │
│                          │                                      │
│                          ▼                                      │
│                   ┌──────────────┐                             │
│                   │ OpenAI兼容API│                             │
│                   │ • 数据增强   │                             │
│                   │ • 检索注入   │                             │
│                   └──────────────┘                             │
│                                                                 │
│   数据提供方收益: 数据变现 + 完全可控 + 隐私保护                 │
│   大模型厂商收益: 高质量数据 + 合规使用 + 零迁移成本             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**核心能力**:
1. **数据可用不可见** - TEE内处理，原始数据不离开安全区
2. **OpenAI兼容** - 标准接口，零迁移成本
3. **零知识证明** - 证明数据质量，不泄露内容
4. **使用审计** - 谁用了什么数据，用了多少次

---

## 2. 极简架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        接入层 (API Gateway)                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  OpenAI兼容接口                                          │  │
│   │  • POST /v1/chat/completions                            │  │
│   │  • POST /v1/embeddings                                  │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      可信执行层 (TEE)                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  数据保险箱 (Data Vault)                                 │  │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐                │  │
│   │  │ 数据A   │  │ 数据B   │  │ 数据C   │  ...           │  │
│   │  │(加密存储)│  │(加密存储)│  │(加密存储)│                │  │
│   │  └─────────┘  └─────────┘  └─────────┘                │  │
│   │                                                         │  │
│   │  ┌─────────────────────────────────────────────────┐   │  │
│   │  │  处理引擎 (在TEE内运行)                          │   │  │
│   │  │  • 数据检索      • 数据增强                      │   │  │
│   │  │  • 提示组装      • 结果过滤                      │   │  │
│   │  └─────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      证明层 (ZKP + 区块链)                      │
│   ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐   │
│   │ ZKP证明生成     │  │ 区块链存证      │  │ 使用审计     │   │
│   │ • 数据质量证明  │  │ • 数据哈希上链  │  │ • 调用记录   │   │
│   │ • 数据完整性    │  │ • 证明上链      │  │ • 用量统计   │   │
│   └─────────────────┘  └─────────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. OpenAI兼容接口设计

### 3.1 标准接口

```python
# 与OpenAI API 100%兼容

# 1. 对话补全 (标准)
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "你是金融专家助手"},
    {"role": "user", "content": "分析这份财报的风险点"}
  ],
  "data_sources": ["finance-reports-2024"],  # 指定受保护数据源
  "temperature": 0.7
}

# 2. 文本嵌入 (标准)
POST /v1/embeddings
{
  "model": "text-embedding-3-small",
  "input": "需要分析的文本",
  "data_source": "finance-reports-2024"  # 可选：在特定数据源上计算
}
```

### 3.2 数据增强方式

```python
# 方式1: 自动检索注入 (最常用)
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "分析市场风险"}
  ],
  "data_sources": ["finance-reports-2024", "market-data-2024"],
  "auto_retrieve": true,  # 自动检索相关数据注入上下文
  "top_k": 5  # 注入Top 5相关片段
}

# 系统内部处理:
# 1. TEE内检索与查询相关的数据片段
# 2. 组装增强的system prompt
# 3. 发送到上游大模型
# 4. 返回结果

# 方式2: 显式数据引用
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "基于以下受保护数据回答："},
    {"role": "data", "source": "finance-reports-2024/Q3.pdf", "top_k": 3},
    {"role": "user", "content": "分析营收趋势"}
  ]
}

# 方式3: 数据过滤 (仅使用特定数据)
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "分析银行板块"}
  ],
  "data_sources": ["finance-reports-2024"],
  "data_filter": {
    "industry": "banking",
    "date_range": "2024-Q3"
  }
}
```

---

## 4. 数据可用不可见的实现方式

### 方式1: TEE内计算 (主推)

```
数据提供方          可信数据护盾              大模型厂商
    │                    │                       │
    │  1.上传加密数据    │                       │
    │─────────────────▶  │                       │
    │                    │  2.TEE内解密处理       │
    │                    │  ┌───────────────┐    │
    │                    │  │ Intel SGX     │    │
    │                    │  │ ┌───────────┐ │    │
    │                    │  │ │ 解密数据  │ │    │
    │                    │  │ │ 检索片段  │ │    │
    │                    │  │ │ 组装Prompt│ │    │
    │                    │  │ └───────────┘ │    │
    │                    │  └───────────────┘    │
    │                    │                       │
    │                    │  3.调用上游模型       │
    │                    │─────────────────────▶│
    │                    │  (只传Prompt,不传数据)│
    │                    │                       │
    │                    │  4.返回模型结果       │
    │                    │◀─────────────────────│
    │                    │                       │
    │                    │  5.返回最终结果       │
    │                    │─────────────────────▶│
    │                    │  (原始数据不离开TEE)   │
```

**特点**:
- 数据在TEE内解密和处理
- 输出的是**增强后的Prompt/结果**，不是原始数据
- 硬件级安全保证

### 方式2: 同态加密 (备选，计算密集型)

```
数据提供方          可信数据护盾              大模型厂商
    │                    │                       │
    │  1.上传同态加密数据 │                       │
    │─────────────────▶  │                       │
    │                    │  2.密文上计算          │
    │                    │  (无需解密)            │
    │                    │                       │
    │                    │  3.返回加密结果        │
    │                    │─────────────────────▶│
    │                    │                       │
    │◀───────────────────│  4.大模型方解密使用    │
```

**特点**:
- 数据始终加密
- 计算开销大（慢10-100倍）
- 适合简单计算（如统计）

---

## 5. 核心服务设计

### 5.1 数据保险箱服务

```python
# services/data_vault/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import hashlib

app = FastAPI(title="Data Vault Service")

class DataUploadRequest(BaseModel):
    data_provider_id: str
    dataset_name: str
    encrypted_data: bytes  # 客户端加密的数据
    schema: dict  # 数据结构描述
    access_policy: dict  # 访问控制策略

class DataUploadResponse(BaseModel):
    dataset_id: str
    data_hash: str
    zkp_proof_id: str  # 数据质量证明
    status: str

@app.post("/v1/data/upload")
async def upload_data(req: DataUploadRequest) -> DataUploadResponse:
    """数据上传入口 - 数据在客户端已加密"""
    
    # 1. 验证数据提供者身份
    await verify_provider(req.data_provider_id)
    
    # 2. 计算数据指纹（不解密）
    data_hash = hashlib.sha256(req.encrypted_data).hexdigest()
    
    # 3. 存储到TEE保护的存储
    dataset_id = generate_dataset_id()
    await tee_storage.store(dataset_id, req.encrypted_data)
    
    # 4. 在TEE内生成ZKP证明（证明数据质量）
    proof_id = await generate_zkp_in_tee(dataset_id, req.schema)
    
    # 5. 区块链存证
    await blockchain.record(dataset_id, data_hash, proof_id)
    
    return DataUploadResponse(
        dataset_id=dataset_id,
        data_hash=data_hash,
        zkp_proof_id=proof_id,
        status="active"
    )

@app.get("/v1/data/{dataset_id}/proof")
async def get_data_proof(dataset_id: str) -> DataProof:
    """获取数据ZKP证明 - 供大模型厂商验证"""
    proof = await get_zkp_proof(dataset_id)
    return DataProof(
        proof=proof.proof_bytes,
        public_inputs=proof.public_inputs,
        quality_score=proof.quality_score,
        blockchain_tx=proof.tx_hash
    )
```

### 5.2 OpenAI兼容网关

```python
# services/api_gateway/openai_compatible.py

from fastapi import FastAPI, Request
import httpx

app = FastAPI()

# 上游模型配置
MODEL_ROUTES = {
    "gpt-4": "https://api.openai.com/v1",
    "gpt-3.5-turbo": "https://api.openai.com/v1",
    "claude-3": "https://api.anthropic.com/v1",
}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI兼容的对话接口 - 核心功能"""
    body = await request.json()
    
    # 1. 检查是否需要注入受保护数据
    if "data_sources" in body:
        # 在TEE内检索相关数据片段
        context = await retrieve_in_tee(
            sources=body["data_sources"],
            query=body["messages"][-1]["content"],
            top_k=body.get("top_k", 5),
            filters=body.get("data_filter", {})
        )
        
        # 组装增强的prompt
        enhanced_system = f"""基于以下受保护数据回答问题：

{context}

注意：以上数据来自可信数据源，请基于这些信息回答。"""
        
        # 插入到messages开头
        body["messages"].insert(0, {
            "role": "system",
            "content": enhanced_system
        })
    
    # 2. 路由到上游模型
    model = body.get("model", "gpt-4")
    upstream_url = MODEL_ROUTES.get(model)
    
    if not upstream_url:
        return {"error": "Unsupported model"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{upstream_url}/chat/completions",
            headers={"Authorization": f"Bearer {get_api_key(model)}"},
            json=body
        )
    
    # 3. 记录使用审计
    await audit_log.record(
        user_id=request.headers.get("X-User-ID"),
        model=model,
        data_sources=body.get("data_sources", []),
        input_tokens=response.json()["usage"]["prompt_tokens"],
        output_tokens=response.json()["usage"]["completion_tokens"]
    )
    
    return response.json()

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """文本嵌入接口"""
    body = await request.json()
    
    # 如果指定了数据源，在TEE内计算
    if "data_source" in body:
        # TEE内加载数据并计算嵌入
        embedding = await compute_embedding_in_tee(
            data_source=body["data_source"],
            text=body["input"]
        )
        return {
            "object": "list",
            "data": [{"embedding": embedding}],
            "model": body.get("model", "text-embedding-3-small")
        }
    
    # 否则转发到标准嵌入服务
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json=body
        )
    return response.json()
```

### 5.3 TEE内处理引擎

```rust
// services/tee_engine/src/lib.rs

use sgx_tcrypto::*;
use sgx_types::*;

/// TEE内数据处理器
pub struct TEEProcessor {
    data_vault: DataVault,
}

impl TEEProcessor {
    /// 在TEE内检索数据片段
    pub fn retrieve_data(
        &self,
        dataset_id: &str,
        query: &str,
        top_k: usize,
        filters: &HashMap<String, String>
    ) -> Result<String, TEEError> {
        // 1. 从安全存储加载加密数据
        let encrypted_data = self.data_vault.load(dataset_id)?;
        
        // 2. 在TEE内解密
        let data = self.decrypt_in_enclave(&encrypted_data)?;
        
        // 3. 应用过滤器
        let filtered = self.apply_filters(&data, filters)?;
        
        // 4. 检索相关片段（可以是关键词匹配、语义检索等）
        let fragments = self.search_fragments(&filtered, query, top_k)?;
        
        // 5. 组装成文本（不返回原始数据）
        let context = fragments.iter()
            .map(|f| format!("[片段{}] {}\n", f.id, f.summary))
            .collect::<String>();
        
        Ok(context)
    }
    
    /// 生成ZKP证明
    pub fn generate_zkp_proof(
        &self,
        dataset_id: &str
    ) -> Result<ZKPProof, TEEError> {
        // 1. 加载数据
        let encrypted = self.data_vault.load(dataset_id)?;
        let data = self.decrypt_in_enclave(&encrypted)?;
        
        // 2. 计算质量指标
        let quality_score = self.calculate_quality(&data);
        let data_size = data.len();
        
        // 3. 生成ZKP证明
        let proof = self.zkp_prover.prove(DataQualityCircuit {
            data: &data,  // 私有输入
            quality_score,  // 公开输入
            data_size,  // 公开输入
        })?;
        
        Ok(proof)
    }
    
    /// 计算文本嵌入（在TEE内）
    pub fn compute_embedding(
        &self,
        dataset_id: &str,
        text: &str
    ) -> Result<Vec<f32>, TEEError> {
        // 1. 加载数据
        let encrypted = self.data_vault.load(dataset_id)?;
        let data = self.decrypt_in_enclave(&encrypted)?;
        
        // 2. 在TEE内计算嵌入（使用本地模型）
        let embedding = self.embedding_model.encode(&data, text)?;
        
        Ok(embedding)
    }
}
```

---

## 6. 数据可用不可见的具体形式

### 形式1: 检索片段注入

```
用户查询: "查找Q3营收增长超过20%的公司"

TEE内处理:
  1. 解密财报数据
  2. 分析每家公司的Q3营收
  3. 筛选符合条件的公司
  4. 生成摘要片段

组装成Prompt:
  "基于以下受保护数据回答问题：
  
  [片段1] ABC银行: Q3营收同比增长25%，主要得益于...
  [片段2] XYZ保险: Q3营收同比增长30%，受益于...
  
  用户问题: 查找Q3营收增长超过20%的公司"

发送到GPT-4 → 返回回答

原始财报数据从未离开TEE！
```

### 形式2: 数据增强上下文

```
用户问题: "分析市场风险"

TEE内处理:
  1. 检索相关市场数据
  2. 组装成增强的system prompt
  3. 发送到上游大模型

发送到GPT-4的完整prompt:
  "基于以下受保护的市场数据回答问题：
  
  [市场数据摘要1] 2024年Q3股市波动率...
  [市场数据摘要2] 债券收益率变化...
  
  用户问题: 分析市场风险"

用户看到的:
  GPT-4的回答（基于受保护数据，但看不到原始数据）
```

### 形式3: 聚合统计结果

```
用户查询: "计算行业平均利润率"

TEE内处理:
  1. 加载所有相关公司的财报
  2. 计算平均利润率
  3. 返回统计结果（不返回明细）

组装成Prompt:
  "基于以下统计数据回答问题：
  
  行业平均净利润率: 15.3%
  样本数量: 50家公司
  置信区间: ±2.1%
  
  用户问题: 计算行业平均利润率"

50家公司的详细财务数据从未暴露！
```

### 形式4: 嵌入向量（用于下游任务）

```
用户请求: 计算文本与某数据集的相似度

TEE内处理:
  1. 加载数据集
  2. 在TEE内计算嵌入向量
  3. 返回向量（不包含原始文本）

返回结果:
  {
    "embedding": [0.1, 0.2, ...],  # 1536维向量
    "model": "text-embedding-3-small",
    "data_source": "finance-reports-2024"
  }

原始文本在TEE内处理，只返回向量！
```

---

## 7. 简化后的部署架构

```yaml
# docker-compose.yml

version: '3.8'

services:
  # API网关 (OpenAI兼容)
  api-gateway:
    build: ./services/api_gateway
    ports:
      - "8000:8000"
    environment:
      - TEE_ENDPOINT=http://tee-engine:8080
      - BLOCKCHAIN_ENDPOINT=http://blockchain:8545
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - tee-engine

  # TEE引擎 (核心)
  tee-engine:
    build:
      context: ./services/tee_engine
      dockerfile: Dockerfile.sgx
    devices:
      - /dev/sgx_enclave
    volumes:
      - tee_data:/data:rw  # 加密数据存储
    environment:
      - SGX_MODE=HW
      - DATA_VAULT_PATH=/data

  # 数据保险箱 (元数据管理)
  data-vault:
    build: ./services/data_vault
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/vault
      - TEE_ENDPOINT=http://tee-engine:8080
    depends_on:
      - postgres
      - tee-engine

  # 区块链节点 (轻节点，用于审计)
  blockchain:
    image: hyperledger/fabric-peer:2.4
    volumes:
      - blockchain_data:/var/hyperledger

  # 数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: vault
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  tee_data:
  blockchain_data:
  postgres_data:
```

---

## 8. 核心价值总结

```
┌─────────────────────────────────────────────────────────────────┐
│                     可信数据护盾 = 数据保险箱                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  数据提供方                        大模型厂商/开发者             │
│  ┌──────────┐                    ┌──────────┐                  │
│  │ 原始数据 │───加密上传────────▶│ 数据保险箱 │                 │
│  │          │                    │   (TEE)   │                 │
│  │ 完全控制 │◀───使用审计────────│           │                 │
│  │ 谁用了   │                    │ 可用不可见 │                 │
│  │ 用了多少 │                    │ 安全计算   │                 │
│  └──────────┘                    └─────┬─────┘                 │
│                                        │                        │
│                                        ▼                        │
│                              ┌─────────────────┐               │
│                              │ OpenAI兼容API   │               │
│                              │ • chat/completions│             │
│                              │ • embeddings    │               │
│                              │ • 数据增强      │               │
│                              └─────────────────┘               │
│                                        │                        │
│                                        ▼                        │
│                              ┌─────────────────┐               │
│                              │ GPT-4/Claude等  │               │
│                              │ 上游大模型       │               │
│                              └─────────────────┘               │
│                                                                 │
│  数据可用不可见形式:                                            │
│  • 检索片段注入  • 数据增强上下文  • 聚合统计  • 嵌入向量       │
│                                                                 │
│  价值:                                                          │
│  • 数据提供方: 数据变现 + 完全可控 + 隐私保护                   │
│  • 大模型厂商: 高质量数据 + 合规使用 + 零迁移成本               │
│  • 平台: 可信基础设施 + 审计透明 + 价值分配                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**简化要点**:
1. **无重排功能** - 专注于数据保险箱核心能力
2. **无向量数据库** - 用TEE内计算替代
3. **OpenAI兼容** - 零迁移成本，开发者友好
4. **多种可用不可见形式** - 片段、增强、统计、嵌入
5. **核心就是数据保险箱** - TEE保护 + ZKP证明 + 审计追溯

**作者**: 灵枢 (架构师)  
**日期**: 2026-04-17
