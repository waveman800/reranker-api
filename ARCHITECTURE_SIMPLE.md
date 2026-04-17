# 可信数据护盾 - 简化架构方案

**核心洞察**: 不需要向量数据库，重点是**数据可用不可见** + **OpenAI兼容接口**

---

## 1. 核心定位

```
┌─────────────────────────────────────────────────────────────────┐
│                     可信数据护盾 = 数据保险箱 + API网关           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   数据提供方                          大模型厂商/开发者          │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐            │
│   │ 原始数据 │───▶│  可信数据    │───▶│ 模型训练 │            │
│   │ (加密)   │    │  护盾(TEE)   │    │ /推理    │            │
│   └──────────┘    └──────────────┘    └──────────┘            │
│                          │                                      │
│                          ▼                                      │
│                   ┌──────────────┐                             │
│                   │ OpenAI兼容   │                             │
│                   │ API接口      │                             │
│                   └──────────────┘                             │
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
│   │  • POST /v1/rerank (扩展)                               │  │
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
│   │  │  • 数据检索      • 重排序                        │   │  │
│   │  │  • 数据增强      • 提示组装                      │   │  │
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
│   │ • 计算完整性    │  │ • 证明上链      │  │ • 用量统计   │   │
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
  "model": "trusted-data-shield/finance-v1",
  "messages": [
    {"role": "system", "content": "你是金融专家助手"},
    {"role": "user", "content": "分析这份财报的风险点"}
  ],
  "temperature": 0.7
}

# 2. 文本嵌入 (标准)
POST /v1/embeddings
{
  "model": "trusted-data-shield/embeddings-v1",
  "input": "需要分析的文本"
}

# 3. 数据检索 (扩展，非标准但类似)
POST /v1/data/retrieve
{
  "dataset": "finance-reports-2024",
  "query": "Q3营收增长",
  "top_k": 5,
  "filters": {"industry": "banking"}
}
```

### 3.2 关键创新：数据注入方式

```python
# 方式1: 自动检索注入 (最常用)
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [...],
  "data_sources": ["finance-reports-2024", "market-data-2024"],  # 指定数据源
  "auto_retrieve": true  # 自动检索相关数据注入上下文
}
# 系统内部：TEE内检索 → 组装prompt → 调用模型 → 返回结果

# 方式2: 显式数据引用
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "基于以下财报数据回答："},
    {"role": "data", "source": "finance-reports-2024/Q3.pdf"},  # 引用受保护数据
    {"role": "user", "content": "分析营收趋势"}
  ]
}

# 方式3: 重排序增强 (RAG增强)
POST /v1/rerank  # 我们的特色接口
{
  "query": "查找相关法规",
  "documents": ["doc1", "doc2", "doc3"],  # 文档ID
  "dataset": "legal-regulations-2024",
  "top_k": 3
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
    │                    │  │ │ 检索/排序 │ │    │
    │                    │  │ │ 组装结果  │ │    │
    │                    │  │ └───────────┘ │    │
    │                    │  └───────────────┘    │
    │                    │                       │
    │                    │  3.返回处理结果       │
    │                    │─────────────────────▶│
    │                    │  (原始数据不离开TEE)   │
```

**特点**:
- 数据在TEE内解密和处理
- 输出的是**结果**，不是原始数据
- 硬件级安全保证

### 方式2: 同态加密 (备选)

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
- 适合简单计算

### 方式3: 安全多方计算 (MPC) (备选)

```
数据提供方A          数据提供方B           可信数据护盾
    │                    │                    │
    │ 1.分片上传         │ 1.分片上传          │
    │─────────────────▶  │─────────────────▶  │
    │                    │                    │
    │                    │                    │ 2.MPC协议计算
    │                    │                    │   (各方不泄露)
    │                    │                    │
    │◀───────────────────┼◀───────────────────│ 3.返回结果
```

**特点**:
- 多方数据联合计算
- 无单一信任方
- 通信开销大

---

## 5. 核心服务设计

### 5.1 数据保险箱服务

```python
# services/data_vault/main.py

from fastapi import FastAPI, Depends
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
    "claude-3": "https://api.anthropic.com/v1",
    "local-llm": "http://localhost:8000/v1"
}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI兼容的对话接口"""
    body = await request.json()
    
    # 1. 检查是否需要注入受保护数据
    if "data_sources" in body:
        # 在TEE内检索相关数据
        context = await retrieve_in_tee(
            sources=body["data_sources"],
            query=body["messages"][-1]["content"],
            top_k=5
        )
        
        # 组装增强的prompt
        body["messages"].insert(0, {
            "role": "system",
            "content": f"基于以下受保护数据回答：\n{context}"
        })
    
    # 2. 路由到上游模型
    model = body.get("model", "gpt-4")
    upstream_url = MODEL_ROUTES.get(model)
    
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

@app.post("/v1/data/retrieve")
async def data_retrieve(request: DataRetrieveRequest):
    """受保护数据检索接口"""
    
    # 1. 验证访问权限
    await check_access_permission(
        user_id=request.user_id,
        dataset=request.dataset
    )
    
    # 2. 在TEE内执行检索
    results = await search_in_tee(
        dataset=request.dataset,
        query=request.query,
        filters=request.filters,
        top_k=request.top_k
    )
    
    # 3. 返回检索结果（不是原始数据）
    return DataRetrieveResponse(
        results=[
            {
                "id": r.id,
                "relevance_score": r.score,
                "summary": r.summary,  # TEE内生成的摘要
                "metadata": r.metadata
            }
            for r in results
        ],
        total_found=results.total
    )
```

### 5.3 TEE内处理引擎

```rust
// services/tee_engine/src/lib.rs

use sgx_tcrypto::*;
use sgx_types::*;

/// TEE内数据处理器
pub struct TEEProcessor {
    data_vault: DataVault,
    reranker: Reranker,
}

impl TEEProcessor {
    /// 在TEE内检索数据
    pub fn retrieve_data(
        &self,
        dataset_id: &str,
        query: &str,
        top_k: usize
    ) -> Result<Vec<RetrievalResult>, TEEError> {
        // 1. 从安全存储加载加密数据
        let encrypted_data = self.data_vault.load(dataset_id)?;
        
        // 2. 在TEE内解密
        let data = self.decrypt_in_enclave(&encrypted_data)?;
        
        // 3. 执行检索（可以是向量检索、关键词检索等）
        let results = self.search(&data, query, top_k)?;
        
        // 4. 生成摘要（不返回原始数据）
        let summarized = results.iter().map(|r| {
            RetrievalResult {
                id: r.id.clone(),
                score: r.score,
                summary: self.generate_summary(&r.content),  // 摘要
                metadata: r.metadata.clone(),
            }
        }).collect();
        
        Ok(summarized)
    }
    
    /// 在TEE内执行重排序
    pub fn rerank(
        &self,
        query: &str,
        doc_ids: &[String],
        dataset_id: &str
    ) -> Result<Vec<ScoredDocument>, TEEError> {
        // 1. 加载文档
        let mut docs = vec![];
        for id in doc_ids {
            let encrypted = self.data_vault.load_doc(dataset_id, id)?;
            let doc = self.decrypt_in_enclave(&encrypted)?;
            docs.push(doc);
        }
        
        // 2. 执行重排序
        let scores = self.reranker.rerank(query, &docs)?;
        
        // 3. 返回分数（不返回文档内容）
        Ok(scores)
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
        let label_distribution = self.calculate_distribution(&data);
        
        // 3. 生成ZKP证明
        let proof = self.zkp_prover.prove(DataQualityCircuit {
            data: &data,  // 私有输入
            quality_score,  // 公开输入
            label_distribution,  // 公开输入
        })?;
        
        Ok(proof)
    }
}
```

---

## 6. 数据可用不可见的具体形式

### 6.1 形式1: 检索结果（最常用）

```
用户查询: "查找Q3营收增长超过20%的公司"

TEE内处理:
  1. 解密财报数据
  2. 分析每家公司的Q3营收
  3. 筛选符合条件的公司
  4. 生成摘要（不返回完整财报）

返回结果:
  [
    {
      "company": "ABC银行",
      "revenue_growth": "25%",
      "summary": "Q3营收同比增长25%，主要得益于...",
      "source": "finance-reports-2024/ABC-Q3.pdf"
    },
    ...
  ]

原始财报数据从未离开TEE！
```

### 6.2 形式2: 增强Prompt

```
用户问题: "分析市场风险"

TEE内处理:
  1. 检索相关市场数据
  2. 组装成增强的system prompt
  3. 发送到上游大模型

发送到GPT-4的prompt:
  "基于以下受保护的市场数据回答问题：
   
   [市场数据摘要1]
   [市场数据摘要2]
   ...
   
   用户问题: 分析市场风险"

用户看到的:
  GPT-4的回答（基于受保护数据，但看不到原始数据）
```

### 6.3 形式3: 重排序分数

```
用户已有文档列表，需要排序

POST /v1/rerank
{
  "query": "相关法规",
  "documents": ["doc1", "doc2", "doc3"],
  "dataset": "legal-regulations-2024"
}

TEE内处理:
  1. 加载doc1, doc2, doc3的完整内容
  2. 计算与"相关法规"的相关性分数
  3. 返回排序后的分数

返回结果:
  {
    "results": [
      {"doc_id": "doc2", "score": 0.95, "rank": 1},
      {"doc_id": "doc1", "score": 0.82, "rank": 2},
      {"doc_id": "doc3", "score": 0.67, "rank": 3}
    ]
  }

文档内容从未离开TEE，只返回分数！
```

### 6.4 形式4: 聚合统计

```
用户查询: "计算行业平均利润率"

TEE内处理:
  1. 加载所有相关公司的财报
  2. 计算平均利润率
  3. 返回统计结果

返回结果:
  {
    "metric": "平均净利润率",
    "value": "15.3%",
    "sample_size": 50,
    "confidence_interval": "±2.1%"
  }

50家公司的详细财务数据从未暴露！
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

  # 区块链节点 (轻节点)
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
│  数据提供方                        大模型厂商                    │
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
│                              │ • data/retrieve │               │
│                              └─────────────────┘               │
│                                        │                        │
│                                        ▼                        │
│                              ┌─────────────────┐               │
│                              │ GPT-4/Claude等  │               │
│                              │ 上游大模型       │               │
│                              └─────────────────┘               │
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
1. **无向量数据库** - 用TEE内计算替代，更通用
2. **OpenAI兼容** - 零迁移成本，开发者友好
3. **多种可用不可见形式** - 检索、增强、排序、统计
4. **核心就是数据保险箱** - TEE保护 + ZKP证明 + 审计追溯

**作者**: 灵枢 (架构师)  
**日期**: 2026-04-17
