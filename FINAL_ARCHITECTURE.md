# 可信数据护盾 - 最终定稿技术方案

**版本**: v1.0 (Final)  
**日期**: 2026-04-17  
**状态**: 经三轮全员讨论定稿  
**团队**: 灵枢(架构师)、小智(PM)、渊码(全栈)、鉴心(QA)

---

## 执行摘要

经过三轮全员讨论，我们确定**可信数据护盾**的核心定位：

> **数据保险箱 + OpenAI兼容接口 = 让高价值数据安全流通**

**关键决策**:
- ❌ 不做重排序服务（与核心定位偏离）
- ❌ 不强制向量数据库（增加复杂度）
- ✅ 专注TEE数据保险箱（硬件级安全）
- ✅ 100% OpenAI兼容（零迁移成本）
- ✅ 零知识证明（证明质量不泄露内容）

---

## 第一轮讨论：核心定位澄清

### 讨论要点

**灵枢**: 我们要解决的核心问题是什么？

**小智**: 数据拥有方（金融/医疗/法律）不敢共享数据，大模型厂商拿不到高质量数据。信任鸿沟。

**渊码**: 技术上就是**数据可用不可见**。TEE是最佳方案，硬件级保证。

**鉴心**: 必须可验证。ZKP证明数据质量，区块链审计使用记录。

### 第一轮结论

```
┌─────────────────────────────────────────────────────────────────┐
│                     核心定位：数据保险箱                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  问题: 数据拥有方不敢共享 ↔ 大模型厂商拿不到数据                │
│                                                                 │
│  解决方案:                                                      │
│  ┌──────────┐      ┌──────────────┐      ┌──────────┐        │
│  │ 原始数据 │─────▶│   TEE保险箱   │─────▶│ 安全使用 │        │
│  │ (加密)   │      │ (可用不可见)  │      │ (有审计) │        │
│  └──────────┘      └──────────────┘      └──────────┘        │
│                                                                 │
│  关键能力:                                                      │
│  • 数据在TEE内解密处理，原始数据不流出                          │
│  • ZKP证明数据质量，不泄露内容                                  │
│  • 区块链记录谁用了什么、用了多少                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第二轮讨论：接口设计决策

### 讨论要点

**小智**: 大模型厂商最关心什么？

**渊码**: 接入成本。如果接口不兼容，他们不愿意改代码。

**灵枢**: 那就做OpenAI 100%兼容。零迁移成本。

**鉴心**: 如何注入受保护数据？要透明。

**渊码**: 通过`data_sources`参数，系统自动在TEE内检索注入。

### 第二轮结论

**接口设计原则**:
1. **100% OpenAI兼容** - 不改一行代码
2. **数据注入透明** - 通过参数指定数据源
3. **输出标准格式** - 用户无感知

```python
# 标准OpenAI调用，只增加data_sources参数
POST /v1/chat/completions
{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "分析风险"}],
  "data_sources": ["finance-reports-2024"],  # 新增：指定受保护数据源
  "top_k": 5  # 新增：注入Top 5相关片段
}

# 返回标准OpenAI格式
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "基于财报数据分析，风险点包括..."
    }
  }]
}
```

---

## 第三轮讨论：技术栈与部署

### 讨论要点

**渊码**: TEE选什么？SGX还是SEV？

**灵枢**: Intel SGX生态更成熟，云厂商支持好。先SGX，后续支持SEV。

**鉴心**: ZKP选什么方案？

**渊码**: Groth16成熟，证明小，适合上链。用circom+snarkjs。

**小智**: 私有化部署怎么支持？

**渊码**: Docker Compose单节点 + K8s集群版。企业可以 air-gap 部署。

### 第三轮结论

**技术栈**:
- **TEE**: Intel SGX (Gramine/Occlum)
- **ZKP**: Groth16 (circom + snarkjs)
- **区块链**: Hyperledger Fabric (联盟链，可控)
- **API**: FastAPI (Python) + Rust (TEE内)
- **部署**: Docker Compose / Kubernetes

---

## 最终架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              可信数据护盾 v1.0                                   │
│                         (经三轮全员讨论定稿)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         接入层 (API Gateway)                            │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │                    OpenAI 100% 兼容接口                          │   │   │
│  │  │                                                                 │   │   │
│  │  │  POST /v1/chat/completions                                      │   │   │
│  │  │  POST /v1/embeddings                                            │   │   │
│  │  │                                                                 │   │   │
│  │  │  扩展参数:                                                      │   │   │
│  │  │    • data_sources: ["dataset-id-1", "dataset-id-2"]            │   │   │
│  │  │    • top_k: 5                    # 注入片段数量                │   │   │
│  │  │    • data_filter: {...}          # 数据过滤条件                │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      可信执行层 (TEE - Intel SGX)                       │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │                     数据保险箱 (Data Vault)                      │   │   │
│  │  │                                                                 │   │   │
│  │  │   加密数据存储:                                                  │   │   │
│  │  │   ┌─────────┐  ┌─────────┐  ┌─────────┐                      │   │   │
│  │  │   │ 金融数据 │  │ 医疗数据 │  │ 法律数据 │  ...                │   │   │
│  │  │   │(AES-256)│  │(AES-256)│  │(AES-256)│                      │   │   │
│  │  │   └─────────┘  └─────────┘  └─────────┘                      │   │   │
│  │  │                                                                 │   │   │
│  │  │   ┌─────────────────────────────────────────────────────────┐  │   │   │
│  │  │   │              TEE内处理引擎 (Rust)                        │  │   │   │
│  │  │   │                                                         │  │   │   │
│  │  │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │  │   │   │
│  │  │   │  │ 数据解密    │─▶│ 片段检索    │─▶│ 摘要生成    │    │  │   │   │
│  │  │   │  │ (SGX密钥)   │  │ (关键词/语义)│  │ (TEE内)     │    │  │   │   │
│  │  │   │  └─────────────┘  └─────────────┘  └─────────────┘    │  │   │   │
│  │  │   │                                                         │  │   │   │
│  │  │   │  ┌─────────────────────────────────────────────────┐   │  │   │   │
│  │  │   │  │ 输出: 相关片段摘要 (原始数据不离开TEE)          │   │  │   │   │
│  │  │   │  │                                                 │   │  │   │   │
│  │  │   │  │ "[片段1] ABC银行Q3营收增长25%，主要..."        │   │  │   │   │
│  │  │   │  │ "[片段2] XYZ保险净利润同比提升30%..."          │   │  │   │   │
│  │  │   │  └─────────────────────────────────────────────────┘   │  │   │   │
│  │  │   └─────────────────────────────────────────────────────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         证明层 (ZKP + 区块链)                           │   │
│  │                                                                         │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │   │
│  │  │   ZKP证明生成       │  │   区块链存证        │  │   使用审计   │  │   │
│  │  │   (circom+snarkjs)  │  │   (Hyperledger)     │  │   (不可篡改) │  │   │
│  │  │                     │  │                     │  │              │  │   │
│  │  │ • 数据质量证明      │  │ • 数据哈希上链      │  │ • 调用记录   │  │   │
│  │  │ • 数据完整性证明    │  │ • ZKP证明上链       │  │ • 用量统计   │  │   │
│  │  │ • 计算完整性证明    │  │ • 使用记录上链      │  │ • 计费依据   │  │   │
│  │  └─────────────────────┘  └─────────────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 数据流架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           数据流：从上传到使用                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  阶段1: 数据上传 (数据提供方)                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  数据提供方                    可信数据护盾                                      │
│  ┌──────────┐                ┌──────────────┐                                  │
│  │ 原始数据 │───1.加密──────▶│  接收服务    │                                  │
│  │          │   (客户端)     │  (Python)    │                                  │
│  └──────────┘                └──────┬───────┘                                  │
│                                     │                                           │
│                                     ▼                                           │
│                              ┌──────────────┐                                  │
│                              │  TEE内处理   │                                  │
│                              │  ┌────────┐  │                                  │
│                              │  │ 解密   │  │                                  │
│                              │  │ 计算质量│  │                                  │
│                              │  │ 生成ZKP│  │                                  │
│                              │  └────────┘  │                                  │
│                              └──────┬───────┘                                  │
│                                     │                                           │
│                              ┌──────┴───────┐                                  │
│                              │  加密存储    │                                  │
│                              │  (AES-256)   │                                  │
│                              └──────────────┘                                  │
│                                     │                                           │
│                              ┌──────┴───────┐                                  │
│                              │  区块链存证  │                                  │
│                              │  (哈希+证明) │                                  │
│                              └──────────────┘                                  │
│                                                                                 │
│  阶段2: 数据使用 (大模型厂商)                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  大模型厂商                    可信数据护盾                    上游大模型       │
│  ┌──────────┐                ┌──────────────┐                ┌──────────┐     │
│  │ API调用  │───────────────▶│  API网关     │                │          │     │
│  │          │  data_sources  │  (FastAPI)   │                │  GPT-4   │     │
│  └──────────┘                └──────┬───────┘                │  Claude  │     │
│                                     │                        │  ...     │     │
│                                     ▼                        └────┬─────┘     │
│                              ┌──────────────┐                     │           │
│                              │  TEE内处理   │                     │           │
│                              │  ┌────────┐  │                     │           │
│                              │  │ 解密   │  │                     │           │
│                              │  │ 检索   │  │                     │           │
│                              │  │ 组装   │  │                     │           │
│                              │  └────────┘  │                     │           │
│                              └──────┬───────┘                     │           │
│                                     │                             │           │
│                              ┌──────┴───────┐                     │           │
│                              │  增强Prompt  │────────────────────▶│           │
│                              │  (片段摘要)  │  标准OpenAI调用      │           │
│                              └──────────────┘                     │           │
│                                                                     │           │
│                              ┌──────────────┐◀────────────────────┘           │
│                              │  返回结果    │  标准OpenAI响应                  │
│                              └──────┬───────┘                                  │
│                                     │                                           │
│                              ┌──────┴───────┐                                  │
│                              │  记录审计    │                                  │
│                              │  (区块链)    │                                  │
│                              └──────────────┘                                  │
│                                                                                 │
│  阶段3: 结算与审计 (平台运营)                                                    │
│  ═══════════════════════════                                                    │
│                                                                                 │
│  区块链 ──▶ 查询使用记录 ──▶ 生成账单 ──▶ 自动分账                               │
│                                                                                 │
│  数据提供方: 60-70%                                                            │
│  平台运营方: 20-25%                                                            │
│  技术提供方: 5-10%                                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 技术栈详情

### 核心组件

| 层级 | 组件 | 技术选型 | 理由 |
|------|------|----------|------|
| **接入层** | API网关 | FastAPI + Kong | Python生态，高性能 |
| **可信层** | TEE运行时 | Gramine/Occlum | SGX成熟方案 |
| **可信层** | TEE内引擎 | Rust | 内存安全，性能 |
| **证明层** | ZKP电路 | circom | 生态完善 |
| **证明层** | ZKP操作 | snarkjs / bellman | 证明生成/验证 |
| **证明层** | 区块链 | Hyperledger Fabric | 联盟链，可控 |
| **存储层** | 元数据 | PostgreSQL | 成熟可靠 |
| **存储层** | 加密数据 | 文件系统/MinIO | 简单高效 |

### 接口规范

```yaml
# OpenAI兼容接口规范

openapi: 3.0.0
info:
  title: Trusted Data Shield API
  version: 1.0.0
  description: OpenAI兼容的数据保险箱服务

paths:
  /v1/chat/completions:
    post:
      summary: 对话补全（支持数据增强）
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                model:
                  type: string
                  example: "gpt-4"
                messages:
                  type: array
                  items:
                    type: object
                    properties:
                      role:
                        type: string
                        enum: [system, user, assistant]
                      content:
                        type: string
                # 扩展参数
                data_sources:
                  type: array
                  items:
                    type: string
                  description: "受保护数据源ID列表"
                  example: ["finance-reports-2024"]
                top_k:
                  type: integer
                  default: 5
                  description: "注入的相关片段数量"
                data_filter:
                  type: object
                  description: "数据过滤条件"
              required: [model, messages]
      responses:
        200:
          description: 标准OpenAI响应格式
          content:
            application/json:
              schema:
                $ref: "https://api.openai.com/schemas/chat-completion.json"

  /v1/data/upload:
    post:
      summary: 上传受保护数据（数据提供方）
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                encrypted_data:
                  type: string
                  format: binary
                schema:
                  type: object
                  description: "数据结构描述"
                access_policy:
                  type: object
                  description: "访问控制策略"
      responses:
        200:
          description: 上传成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  dataset_id:
                    type: string
                  data_hash:
                    type: string
                  zkp_proof_id:
                    type: string

  /v1/data/{dataset_id}/proof:
    get:
      summary: 获取数据ZKP证明（供验证）
      responses:
        200:
          description: ZKP证明
          content:
            application/json:
              schema:
                type: object
                properties:
                  proof:
                    type: string
                    description: "Base64编码的证明"
                  public_inputs:
                    type: array
                    items:
                      type: string
                  quality_score:
                    type: number
                  blockchain_tx:
                    type: string
```

---

## 部署架构

### 开发环境

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  api-gateway:
    build: ./services/api_gateway
    ports:
      - "8000:8000"
    environment:
      - TEE_MODE=simulation  # 开发模式：模拟TEE
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/tds
    volumes:
      - ./services/api_gateway:/app

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: tds
    ports:
      - "5432:5432"
```

### 生产环境（单节点）

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api-gateway:
    build: ./services/api_gateway
    ports:
      - "80:8000"
      - "443:8000"
    environment:
      - TEE_ENDPOINT=http://tee-engine:8080
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/tds
      - REDIS_URL=redis://redis:6379
    depends_on:
      - tee-engine
      - postgres
      - redis

  tee-engine:
    build:
      context: ./services/tee_engine
      dockerfile: Dockerfile.sgx
    devices:
      - /dev/sgx_enclave
      - /dev/sgx_provision
    environment:
      - SGX_MODE=HW
      - DATA_VAULT_PATH=/data
    volumes:
      - tee_data:/data:rw

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: tds
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  blockchain:
    image: hyperledger/fabric-peer:2.4
    volumes:
      - blockchain_data:/var/hyperledger

volumes:
  tee_data:
  postgres_data:
  redis_data:
  blockchain_data:
```

### 生产环境（Kubernetes）

```yaml
# k8s/tee-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tee-engine
  namespace: trusted-data-shield
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tee-engine
  template:
    metadata:
      labels:
        app: tee-engine
    spec:
      nodeSelector:
        sgx-enabled: "true"
      containers:
      - name: tee
        image: trusted-data-shield/tee-engine:v1.0.0
        resources:
          limits:
            sgx.k8s.io/sgx: "1"
            memory: "4Gi"
            cpu: "2"
        securityContext:
          privileged: true
        volumeMounts:
        - name: sgx-device
          mountPath: /dev/sgx
        - name: tee-data
          mountPath: /data
      volumes:
      - name: sgx-device
        hostPath:
          path: /dev/sgx
      - name: tee-data
        persistentVolumeClaim:
          claimName: tee-data-pvc
```

---

## 实施路线图

### Phase 1: MVP (6周)

| 周 | 任务 | 负责人 | 产出 |
|----|------|--------|------|
| 1 | TEE环境搭建 | 渊码 | SGX开发环境 |
| 1 | ZKP电路设计 | 渊码 | data_quality.circom |
| 2 | 数据保险箱基础 | 渊码 | 上传/存储服务 |
| 2 | API网关 | 渊码 | OpenAI兼容接口 |
| 3 | TEE内检索 | 渊码 | 关键词检索引擎 |
| 3 | ZKP集成 | 渊码 | 证明生成/验证 |
| 4 | 区块链存证 | 渊码 | 使用记录上链 |
| 4 | 审计系统 | 渊码 | 查询接口 |
| 5 | 集成测试 | 鉴心 | 测试报告 |
| 5 | 安全测试 | 鉴心 | 渗透测试报告 |
| 6 | 文档完善 | 小智 | 用户文档 |
| 6 | 内部演示 | 灵枢 | Demo演示 |

### Phase 2: Beta (6周)

- [ ] 语义检索（Embedding-based）
- [ ] 多数据源联合查询
- [ ] 高级访问控制（ABAC）
- [ ] 企业级监控告警
- [ ] 私有化部署工具
- [ ] 客户POC支持

### Phase 3: V1 (6周)

- [ ] AMD SEV支持
- [ ] 联邦学习集成
- [ ] 智能合约自动分账
- [ ] 多区域部署
- [ ] 合规报告生成
- [ ] 生产发布

---

## 团队分工

| 角色 | 负责人 | 核心职责 |
|------|--------|----------|
| **架构师** | 灵枢 | 技术决策、架构设计、跨团队协调 |
| **产品经理** | 小智 | 需求定义、用户调研、文档撰写 |
| **全栈工程师** | 渊码 | 核心开发、TEE/ZKP实现、部署 |
| **测试专家** | 鉴心 | 测试策略、安全测试、质量保障 |

---

## 关键决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 核心定位 | 重排服务 vs 数据保险箱 | **数据保险箱** | 解决信任鸿沟，市场空间更大 |
| 接口标准 | 自定义 vs OpenAI兼容 | **OpenAI兼容** | 零迁移成本，开发者友好 |
| TEE方案 | SGX vs SEV vs TrustZone | **SGX** | 生态成熟，云厂商支持好 |
| ZKP方案 | Groth16 vs STARKs | **Groth16** | 证明小，适合上链 |
| 区块链 | 公链 vs 联盟链 | **Hyperledger** | 可控，符合企业需求 |
| 部署模式 | SaaS vs 私有化 | **都支持** | ToB需求多样化 |

---

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| SGX侧信道攻击 | 中 | 高 | 定期更新微码，使用防护库 |
| ZKP电路漏洞 | 低 | 高 | 形式化验证，第三方审计 |
| 性能瓶颈 | 中 | 中 | GPU加速，缓存优化 |
| 客户接受度 | 中 | 高 | POC验证，免费试用 |
| 合规风险 | 低 | 高 | 法务审核，合规报告 |

---

## 总结

**可信数据护盾**经过三轮全员讨论，最终确定：

> **核心 = 数据保险箱 + OpenAI兼容接口**

**我们不做**:
- ❌ 重排序服务
- ❌ 向量数据库强制依赖
- ❌ 复杂的自定义接口

**我们专注**:
- ✅ TEE硬件级数据保护
- ✅ 零知识证明数据质量
- ✅ 100% OpenAI兼容（零迁移）
- ✅ 区块链审计与分账

**目标**: 让金融、医疗、法律等高价值数据安全流通，让大模型厂商放心使用。

---

**文档状态**: Final v1.0  
**最后更新**: 2026-04-17  
**下次评审**: Phase 1完成后
