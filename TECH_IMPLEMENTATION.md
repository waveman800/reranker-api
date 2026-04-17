# 可信数据护盾 - 技术实现方案

**作者**: 渊码 (全栈工程师)  
**审核**: 灵枢 (架构师)  
**日期**: 2026-04-17  
**版本**: v1.0

---

## 1. 技术架构总览

### 1.1 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         可信数据护盾平台                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        接入层 (Access Layer)                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │ 数据提供方   │  │ 大模型厂商   │  │ 平台运营方   │                │   │
│  │  │   Portal    │  │   API/SDK   │  │   Console   │                │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      服务层 (Service Layer)                         │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │ 数据接入服务 │  │ ZKP证明服务  │  │ 重排序服务   │                │   │
│  │  │  (Python)   │  │  (Rust)     │  │  (Python)   │                │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │ 反蒸馏服务   │  │ 审计追溯服务 │  │ 计费服务    │                │   │
│  │  │  (Python)   │  │  (Go)       │  │  (Go)       │                │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ZKP层 (Zero-Knowledge Layer)                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              ZKP Prover/Verifier (Rust/circom)              │   │   │
│  │  │                                                             │   │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│   │   │
│  │  │  │ 数据质量    │  │ 计算完整性  │  │ 成员证明           ││   │   │
│  │  │  │ 证明电路    │  │ 证明电路    │  │ 电路               ││   │   │
│  │  │  │ (circom)   │  │ (circom)   │  │ (circom)          ││   │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘│   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TEE层 (Trusted Execution)                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Intel SGX / AMD SEV Enclaves                   │   │   │
│  │  │                                                             │   │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│   │   │
│  │  │  │ ZKP验证器   │  │ Reranker    │  │ 反蒸馏引擎         ││   │   │
│  │  │  │ (Rust)     │  │ 模型        │  │ (Rust)            ││   │   │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘│   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      数据层 (Data Layer)                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │ PostgreSQL  │  │  Redis      │  │  MinIO      │                │   │
│  │  │ (元数据)    │  │  (缓存)     │  │  (对象存储)  │                │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Hyperledger Fabric (区块链)                    │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ZKP技术实现

### 2.1 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| **ZKP框架** | circom + snarkjs | 成熟、生态完善、性能好 |
| **证明系统** | Groth16 | 证明最小、验证最快 |
| **椭圆曲线** | BN128 | 以太坊兼容、工具成熟 |
| **编程语言** | Rust (核心) + JavaScript (工具) | 性能 + 生态 |

### 2.2 数据质量证明电路

```circom
// data_quality.circom
// 数据质量证明电路 - 证明数据质量指标不泄露原始数据

template DataQualityProof(n) {
    // 输入信号（私有）
    signal private input data[n];           // 原始数据（不公开）
    signal private input labels[n];         // 标签（不公开）
    
    // 公开输入
    signal input data_hash;                 // 数据指纹（公开）
    signal input quality_score;             // 质量分数（公开）
    signal input label_distribution[10];    // 标签分布（公开）
    
    // 中间信号
    signal computed_hash;
    signal computed_score;
    signal computed_distribution[10];
    
    // 1. 验证数据哈希
    component hasher = Poseidon(n);
    for (var i = 0; i < n; i++) {
        hasher.inputs[i] <== data[i];
    }
    computed_hash <== hasher.out;
    
    // 约束：计算出的哈希必须等于公开的哈希
    computed_hash === data_hash;
    
    // 2. 计算质量分数（简化示例）
    component score_calc = QualityCalculator(n);
    score_calc.data <== data;
    score_calc.labels <== labels;
    computed_score <== score_calc.score;
    
    // 约束：质量分数在合理范围
    computed_score === quality_score;
    
    // 3. 计算标签分布
    component dist_calc = DistributionCalculator(n, 10);
    dist_calc.labels <== labels;
    for (var i = 0; i < 10; i++) {
        computed_distribution[i] <== dist_calc.distribution[i];
        computed_distribution[i] === label_distribution[i];
    }
}

// 质量计算器组件
template QualityCalculator(n) {
    signal input data[n];
    signal input labels[n];
    signal output score;
    
    // 计算数据完整性
    signal completeness;
    component completeness_calc = CompletenessCalculator(n);
    completeness_calc.data <== data;
    completeness <== completeness_calc.result;
    
    // 计算标签平衡性
    signal balance;
    component balance_calc = BalanceCalculator(n);
    balance_calc.labels <== labels;
    balance <== balance_calc.result;
    
    // 综合质量分数（加权）
    score <== completeness * 0.6 + balance * 0.4;
}

// 主电路
component main = DataQualityProof(1000);
```

### 2.3 ZKP服务实现

```rust
// zkp-service/src/lib.rs

use bellman::groth16::{Proof, VerifyingKey, create_random_proof, verify_proof, prepare_verifying_key};
use bellman::{Circuit, ConstraintSystem, SynthesisError};
use bls12_381::Bls12;
use rand::rngs::OsRng;

pub struct ZKPService {
    proving_key: ProvingKey<Bls12>,
    verifying_key: VerifyingKey<Bls12>,
    prepared_vk: PreparedVerifyingKey<Bls12>,
}

impl ZKPService {
    pub fn new(params_path: &str) -> Result<Self, ZKPError> {
        // 加载电路参数
        let proving_key = load_proving_key(params_path)?;
        let verifying_key = load_verifying_key(params_path)?;
        let prepared_vk = prepare_verifying_key(&verifying_key);
        
        Ok(Self {
            proving_key,
            verifying_key,
            prepared_vk,
        })
    }
    
    /// 生成数据质量证明
    pub fn prove_data_quality(
        &self,
        data: &[DataRecord],
        labels: &[Label],
    ) -> Result<DataQualityProof, ZKPError> {
        // 1. 计算数据哈希（公开）
        let data_hash = compute_data_hash(data);
        
        // 2. 计算质量分数（公开）
        let quality_score = calculate_quality_score(data, labels);
        
        // 3. 计算标签分布（公开）
        let label_distribution = calculate_label_distribution(labels);
        
        // 4. 构建电路输入
        let circuit = DataQualityCircuit {
            data: data.to_vec(),                    // 私有输入
            labels: labels.to_vec(),                // 私有输入
            data_hash,                              // 公开输入
            quality_score,                          // 公开输入
            label_distribution,                     // 公开输入
        };
        
        // 5. 生成证明
        let proof = create_random_proof(circuit, &self.proving_key, &mut OsRng)?;
        
        // 6. 构建证明结果
        Ok(DataQualityProof {
            proof,
            public_inputs: PublicInputs {
                data_hash,
                quality_score,
                label_distribution,
            },
        })
    }
    
    /// 验证数据质量证明
    pub fn verify_data_quality(
        &self,
        proof: &DataQualityProof,
    ) -> Result<bool, ZKPError> {
        // 验证证明
        verify_proof(
            &self.prepared_vk,
            &proof.proof,
            &proof.public_inputs.to_vec(),
        ).map_err(|e| ZKPError::VerificationFailed(e.to_string()))
    }
}

/// 数据质量电路
struct DataQualityCircuit {
    data: Vec<DataRecord>,          // 私有输入
    labels: Vec<Label>,             // 私有输入
    data_hash: Hash,                // 公开输入
    quality_score: Score,           // 公开输入
    label_distribution: Distribution, // 公开输入
}

impl Circuit<Bls12> for DataQualityCircuit {
    fn synthesize<CS: ConstraintSystem<Bls12>>(
        self,
        cs: &mut CS
    ) -> Result<(), SynthesisError> {
        // 分配私有输入
        let data_vars = self.data.iter().map(|d| {
            cs.alloc_private(|| "data", || Ok(d.to_field_element()))
        }).collect::<Result<Vec<_>, _>>()?;
        
        // 分配公开输入
        let data_hash_var = cs.alloc_input(
            || "data_hash",
            || Ok(self.data_hash.to_field_element())
        )?;
        
        // 计算哈希并约束
        let computed_hash = poseidon_hash(cs, &data_vars)?;
        cs.enforce(
            || "hash_check",
            |lc| lc + computed_hash,
            |lc| lc + CS::one(),
            |lc| lc + data_hash_var,
        );
        
        // 计算质量分数并约束
        let computed_score = calculate_quality_score_circuit(cs, &data_vars)?;
        let score_var = cs.alloc_input(
            || "quality_score",
            || Ok(self.quality_score.to_field_element())
        )?;
        cs.enforce(
            || "score_check",
            |lc| lc + computed_score,
            |lc| lc + CS::one(),
            |lc| lc + score_var,
        );
        
        Ok(())
    }
}
```

### 2.4 ZKP与TEE结合

```rust
// tee-zkp/src/lib.rs

use sgx_tcrypto::*;
use sgx_types::*;

/// TEE内的ZKP验证器
pub struct TEEZKPVerifier {
    vk: VerifyingKey,
}

impl TEEZKPVerifier {
    /// 在TEE内验证ZKP证明
    pub fn verify_in_enclave(&self, proof: &Proof) -> Result<bool, sgx_status_t> {
        // 1. 验证证明格式
        self.validate_proof_format(proof)?;
        
        // 2. 在TEE内执行验证（保护验证密钥）
        let result = unsafe {
            sgx_ecc256_verify(
                &proof.signature,
                &proof.public_inputs_hash,
                &self.vk.public_key,
            )
        };
        
        // 3. 生成验证报告
        let attestation = self.generate_attestation(proof, result)?;
        
        Ok(result == SGX_SUCCESS)
    }
    
    /// 生成带签名的验证报告
    fn generate_attestation(
        &self,
        proof: &Proof,
        result: bool,
    ) -> Result<AttestationReport, sgx_status_t> {
        let report_data = sgx_report_data_t {
            d: self.hash_proof_result(proof, result),
        };
        
        let report = unsafe {
            sgx_create_report(
                &self.target_info,
                &report_data,
            )
        }?;
        
        Ok(AttestationReport::from(report))
    }
}

/// ZKP证明包装器（带TEE证明）
pub struct TEEWrappedProof {
    pub zkp_proof: Proof,
    pub tee_attestation: AttestationReport,
    pub verification_result: bool,
}
```

---

## 3. 核心服务实现

### 3.1 数据接入服务

```python
# services/data_ingestion/src/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import hashlib
import asyncio

app = FastAPI(title="Data Ingestion Service")

class DataIngestionRequest(BaseModel):
    data_provider_id: str
    data_type: str  # "text", "image", "multimodal"
    data_location: str  # S3 URL or local path
    metadata: dict
    
class DataIngestionResponse(BaseModel):
    data_id: str
    data_hash: str
    quality_score: float
    status: str

@app.post("/api/v1/data/ingest")
async def ingest_data(request: DataIngestionRequest) -> DataIngestionResponse:
    """数据接入入口"""
    
    # 1. 验证数据提供者身份
    await verify_data_provider(request.data_provider_id)
    
    # 2. 下载数据（异步）
    data = await download_data(request.data_location)
    
    # 3. 计算数据指纹
    data_hash = compute_data_hash(data)
    
    # 4. 质量评估
    quality_score = await assess_quality(data, request.data_type)
    
    # 5. 生成数据ID
    data_id = generate_data_id(data_hash, request.data_provider_id)
    
    # 6. 存储元数据
    await store_metadata(data_id, request, data_hash, quality_score)
    
    # 7. 加密存储数据
    encrypted_location = await encrypt_and_store(data_id, data)
    
    # 8. 触发ZKP证明生成（异步）
    asyncio.create_task(generate_zkp_proof(data_id, data, quality_score))
    
    return DataIngestionResponse(
        data_id=data_id,
        data_hash=data_hash,
        quality_score=quality_score,
        status="processing"
    )

async def generate_zkp_proof(data_id: str, data: bytes, quality_score: float):
    """异步生成ZKP证明"""
    # 调用ZKP服务
    zkp_service = ZKPService()
    proof = await zkp_service.prove_data_quality(data)
    
    # 存储证明
    await store_zkp_proof(data_id, proof)
    
    # 更新状态
    await update_data_status(data_id, "ready")
```

### 3.2 ZKP证明服务

```rust
// services/zkp/src/main.rs

use actix_web::{web, App, HttpServer, HttpResponse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
struct ProveRequest {
    data_id: String,
    proof_type: String,  // "data_quality", "computation_integrity", "membership"
}

#[derive(Serialize)]
struct ProveResponse {
    proof_id: String,
    proof: String,  // Base64 encoded
    public_inputs: Vec<String>,
    status: String,
}

async fn generate_proof(
    req: web::Json<ProveRequest>,
    zkp_service: web::Data<Arc<ZKPService>>,
) -> HttpResponse {
    // 1. 获取数据
    let data = fetch_data(&req.data_id).await;
    
    // 2. 根据类型生成证明
    let proof = match req.proof_type.as_str() {
        "data_quality" => {
            zkp_service.prove_data_quality(&data).await
        }
        "computation_integrity" => {
            zkp_service.prove_computation(&data).await
        }
        "membership" => {
            zkp_service.prove_membership(&data).await
        }
        _ => return HttpResponse::BadRequest().body("Unknown proof type"),
    };
    
    // 3. 存储证明
    let proof_id = store_proof(&proof).await;
    
    HttpResponse::Ok().json(ProveResponse {
        proof_id,
        proof: base64::encode(&proof.to_bytes()),
        public_inputs: proof.public_inputs.iter().map(|i| i.to_string()).collect(),
        status: "success".to_string(),
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // 初始化ZKP服务
    let zkp_service = Arc::new(ZKPService::new("/params").unwrap());
    
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(zkp_service.clone()))
            .route("/api/v1/zkp/prove", web::post().to(generate_proof))
            .route("/api/v1/zkp/verify", web::post().to(verify_proof))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
```

### 3.3 反蒸馏服务

```python
# services/anti_distillation/src/engine.py

import numpy as np
from typing import List, Dict, Any
import hashlib
import hmac

class AntiDistillationEngine:
    """反蒸馏保护引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.noise_level = config.get('noise_level', 0.01)
        self.watermark_key = config.get('watermark_key')
        self.fingerprint_enabled = config.get('fingerprint_enabled', True)
    
    def protect(self, data: Any, user_id: str, session_id: str) -> ProtectedData:
        """多层保护"""
        # 1. 输出混淆
        obfuscated = self._obfuscate(data)
        
        # 2. 嵌入水印
        watermarked = self._embed_watermark(obfuscated, user_id, session_id)
        
        # 3. 生成行为指纹
        fingerprint = None
        if self.fingerprint_enabled:
            fingerprint = self._generate_fingerprint(data, user_id)
        
        return ProtectedData(
            data=watermarked,
            fingerprint=fingerprint,
            protection_log={
                'timestamp': datetime.utcnow(),
                'user_id': user_id,
                'session_id': session_id,
                'methods': ['obfuscation', 'watermark', 'fingerprint']
            }
        )
    
    def _obfuscate(self, data: Any) -> Any:
        """输出混淆"""
        if isinstance(data, list):  # 分数列表
            noise = np.random.normal(0, self.noise_level, len(data))
            obfuscated = [max(0, min(1, s + n)) for s, n in zip(data, noise)]
            return self._preserve_ranking(obfuscated, data)
        return data
    
    def _preserve_ranking(self, obfuscated: List[float], original: List[float]) -> List[float]:
        """保持排序一致性"""
        # 确保相对顺序不变
        original_order = sorted(range(len(original)), key=lambda i: original[i], reverse=True)
        current_order = sorted(range(len(obfuscated)), key=lambda i: obfuscated[i], reverse=True)
        
        if original_order != current_order:
            # 调整以恢复顺序
            for i, (orig_idx, curr_idx) in enumerate(zip(original_order, current_order)):
                if orig_idx != curr_idx:
                    # 添加微小差异
                    obfuscated[curr_idx] += (i + 1) * 1e-7
        
        return obfuscated
    
    def _embed_watermark(self, data: Any, user_id: str, session_id: str) -> Any:
        """嵌入水印"""
        # 生成用户特定水印
        watermark = hmac.new(
            self.watermark_key.encode(),
            f"{user_id}:{session_id}".encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        
        # 在数据中嵌入水印（LSB或语义水印）
        if isinstance(data, str):
            return self._embed_text_watermark(data, watermark)
        elif isinstance(data, list):
            return self._embed_score_watermark(data, watermark)
        
        return data
    
    def _embed_score_watermark(self, scores: List[float], watermark: str) -> List[float]:
        """在分数中嵌入水印"""
        # 将水印编码到分数的微小变化中
        watermarked = scores.copy()
        watermark_bits = bin(int(watermark, 16))[2:].zfill(64)
        
        for i, bit in enumerate(watermark_bits[:len(scores)]):
            if bit == '1':
                watermarked[i] += 1e-6  # 微小增加
            else:
                watermarked[i] -= 1e-6  # 微小减少
        
        return watermarked
    
    def _generate_fingerprint(self, data: Any, user_id: str) -> Dict:
        """生成行为指纹"""
        return {
            'user_id': user_id,
            'data_pattern_hash': hashlib.sha256(str(data).encode()).hexdigest()[:16],
            'timestamp': datetime.utcnow().isoformat(),
            'request_count': 1,
            'query_complexity': self._calculate_complexity(data),
        }
    
    def detect_distillation(self, model_output: Any) -> DetectionResult:
        """检测蒸馏行为"""
        result = DetectionResult()
        
        # 1. 水印检测
        watermark_match = self._detect_watermark(model_output)
        if watermark_match:
            result.add_evidence('watermark', watermark_match)
        
        # 2. 行为分析
        behavior_anomaly = self._analyze_behavior(model_output)
        if behavior_anomaly:
            result.add_evidence('behavior', behavior_anomaly)
        
        # 3. 相似度分析
        similarity = self._check_similarity(model_output)
        if similarity > 0.9:  # 阈值
            result.add_evidence('similarity', similarity)
        
        result.is_distillation = len(result.evidence) >= 2
        return result
```

---

## 4. SDK设计

### 4.1 Python SDK

```python
# sdk/python/trusted_data_shield/__init__.py

"""可信数据护盾 Python SDK"""

__version__ = "1.0.0"

from .client import TDSClient
from .data_provider import DataProvider
from .model_consumer import ModelConsumer

__all__ = ['TDSClient', 'DataProvider', 'ModelConsumer']
```

```python
# sdk/python/trusted_data_shield/client.py

from typing import Optional, Dict, Any
import requests
import hashlib

class TDSClient:
    """可信数据护盾客户端"""
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.trusteddata.shield"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
    
    def upload_data(self, data: bytes, metadata: Dict[str, Any]) -> str:
        """上传数据"""
        # 计算数据哈希
        data_hash = hashlib.sha256(data).hexdigest()
        
        # 生成请求签名
        signature = self._sign_request(data_hash)
        
        # 上传数据
        response = self.session.post(
            f"{self.base_url}/v1/data/upload",
            headers={
                'X-API-Key': self.api_key,
                'X-Signature': signature,
                'X-Data-Hash': data_hash,
            },
            files={'data': data},
            data={'metadata': metadata},
        )
        
        result = response.json()
        return result['data_id']
    
    def request_rerank(self, data_id: str, query: str, 
                       top_k: int = 10) -> RerankResult:
        """请求重排序"""
        response = self.session.post(
            f"{self.base_url}/v1/rerank",
            headers=self._auth_headers(),
            json={
                'data_id': data_id,
                'query': query,
                'top_k': top_k,
            },
        )
        
        return RerankResult.from_dict(response.json())
    
    def verify_zkp_proof(self, proof_id: str) -> bool:
        """验证ZKP证明"""
        response = self.session.get(
            f"{self.base_url}/v1/zkp/verify/{proof_id}",
            headers=self._auth_headers(),
        )
        
        result = response.json()
        return result['valid']
    
    def _sign_request(self, data: str) -> str:
        """生成请求签名"""
        import hmac
        timestamp = str(int(time.time()))
        message = f"{timestamp}:{data}"
        return hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
```

### 4.2 Go SDK

```go
// sdk/go/trusted_data_shield/client.go

package trusted_data_shield

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/hex"
    "time"
)

// Client 可信数据护盾客户端
type Client struct {
    APIKey     string
    APISecret  string
    BaseURL    string
    HTTPClient *http.Client
}

// NewClient 创建客户端
func NewClient(apiKey, apiSecret string) *Client {
    return &Client{
        APIKey:     apiKey,
        APISecret:  apiSecret,
        BaseURL:    "https://api.trusteddata.shield",
        HTTPClient: &http.Client{Timeout: 30 * time.Second},
    }
}

// UploadData 上传数据
func (c *Client) UploadData(data []byte, metadata map[string]interface{}) (string, error) {
    // 计算数据哈希
    hash := sha256.Sum256(data)
    dataHash := hex.EncodeToString(hash[:])
    
    // 生成签名
    signature := c.signRequest(dataHash)
    
    // 构建请求
    req, err := http.NewRequest("POST", c.BaseURL+"/v1/data/upload", bytes.NewReader(data))
    if err != nil {
        return "", err
    }
    
    req.Header.Set("X-API-Key", c.APIKey)
    req.Header.Set("X-Signature", signature)
    req.Header.Set("X-Data-Hash", dataHash)
    
    // 发送请求
    resp, err := c.HTTPClient.Do(req)
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()
    
    // 解析响应
    var result struct {
        DataID string `json:"data_id"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return "", err
    }
    
    return result.DataID, nil
}

// signRequest 生成请求签名
func (c *Client) signRequest(data string) string {
    timestamp := strconv.FormatInt(time.Now().Unix(), 10)
    message := timestamp + ":" + data
    
    h := hmac.New(sha256.New, []byte(c.APISecret))
    h.Write([]byte(message))
    return hex.EncodeToString(h.Sum(nil))
}
```

### 4.3 Java SDK

```java
// sdk/java/src/main/java/com/trusteddata/shield/Client.java

package com.trusteddata.shield;

import okhttp3.*;
import com.google.gson.Gson;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.security.MessageDigest;
import java.util.Base64;

public class TDSClient {
    private final String apiKey;
    private final String apiSecret;
    private final String baseUrl;
    private final OkHttpClient httpClient;
    private final Gson gson;
    
    public TDSClient(String apiKey, String apiSecret) {
        this(apiKey, apiSecret, "https://api.trusteddata.shield");
    }
    
    public TDSClient(String apiKey, String apiSecret, String baseUrl) {
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
        this.baseUrl = baseUrl;
        this.httpClient = new OkHttpClient();
        this.gson = new Gson();
    }
    
    /**
     * 上传数据
     */
    public String uploadData(byte[] data, Map<String, Object> metadata) throws Exception {
        // 计算数据哈希
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] hash = digest.digest(data);
        String dataHash = Base64.getEncoder().encodeToString(hash);
        
        // 生成签名
        String signature = signRequest(dataHash);
        
        // 构建请求
        RequestBody body = RequestBody.create(data, MediaType.parse("application/octet-stream"));
        Request request = new Request.Builder()
            .url(baseUrl + "/v1/data/upload")
            .header("X-API-Key", apiKey)
            .header("X-Signature", signature)
            .header("X-Data-Hash", dataHash)
            .post(body)
            .build();
        
        // 发送请求
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new RuntimeException("Upload failed: " + response);
            }
            
            UploadResponse result = gson.fromJson(response.body().string(), UploadResponse.class);
            return result.getDataId();
        }
    }
    
    /**
     * 生成请求签名
     */
    private String signRequest(String data) throws Exception {
        String timestamp = String.valueOf(System.currentTimeMillis() / 1000);
        String message = timestamp + ":" + data;
        
        Mac mac = Mac.getInstance("HmacSHA256");
        SecretKeySpec secretKeySpec = new SecretKeySpec(apiSecret.getBytes(), "HmacSHA256");
        mac.init(secretKeySpec);
        byte[] signatureBytes = mac.doFinal(message.getBytes());
        
        return Base64.getEncoder().encodeToString(signatureBytes);
    }
}
```

---

## 5. 部署架构

### 5.1 Docker Compose配置

```yaml
# docker-compose.yml

version: '3.8'

services:
  # API网关
  gateway:
    image: kong:3.0
    ports:
      - "8000:8000"
      - "8443:8443"
    environment:
      KONG_DATABASE: "off"
      KONG_DECLARATIVE_CONFIG: /kong/declarative/kong.yml
    volumes:
      - ./kong:/kong/declarative
    depends_on:
      - data-ingestion
      - zkp-service
      - reranker-service

  # 数据接入服务
  data-ingestion:
    build:
      context: ./services/data_ingestion
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/tds
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
    depends_on:
      - postgres
      - redis
      - minio

  # ZKP服务
  zkp-service:
    build:
      context: ./services/zkp
      dockerfile: Dockerfile
    environment:
      - RUST_LOG=info
      - CIRCUIT_PATH=/circuits
    volumes:
      - ./circuits:/circuits:ro
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  # 重排序服务（TEE）
  reranker-service:
    build:
      context: ./services/reranker
      dockerfile: Dockerfile.sgx
    devices:
      - /dev/sgx_enclave
      - /dev/sgx_provision
    environment:
      - SGX_MODE=HW
      - MODEL_PATH=/models
    volumes:
      - ./models:/models:ro

  # 数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: tds
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # 缓存
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  # 对象存储
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  redis_data:
  minio_data:
```

### 5.2 Kubernetes部署

```yaml
# k8s/zkp-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: zkp-service
  namespace: trusted-data-shield
spec:
  replicas: 3
  selector:
    matchLabels:
      app: zkp-service
  template:
    metadata:
      labels:
        app: zkp-service
    spec:
      containers:
      - name: zkp
        image: trusted-data-shield/zkp:v1.0.0
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: circuit-params
          mountPath: /params
          readOnly: true
      volumes:
      - name: circuit-params
        persistentVolumeClaim:
          claimName: zkp-params-pvc

---
# TEE节点部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reranker-tee
  namespace: trusted-data-shield
spec:
  replicas: 2
  selector:
    matchLabels:
      app: reranker-tee
  template:
    metadata:
      labels:
        app: reranker-tee
    spec:
      nodeSelector:
        sgx-enabled: "true"  # SGX节点
      containers:
      - name: reranker
        image: trusted-data-shield/reranker-tee:v1.0.0
        resources:
          limits:
            sgx.k8s.io/sgx: "1"  # SGX资源
        securityContext:
          privileged: true
        volumeMounts:
        - name: sgx-device
          mountPath: /dev/sgx
      volumes:
      - name: sgx-device
        hostPath:
          path: /dev/sgx
```

---

## 6. 开发计划

### Phase 1: MVP (6周)

| 周 | 任务 | 负责人 |
|----|------|--------|
| 1 | ZKP电路设计 (circom) | 渊码 |
| 1 | TEE环境搭建 | 渊码 |
| 2 | ZKP服务开发 (Rust) | 渊码 |
| 2 | 数据接入服务 | 渊码 |
| 3 | reranker-api TEE集成 | 渊码 |
| 3 | 基础反蒸馏保护 | 渊码 |
| 4 | Python SDK | 渊码 |
| 4 | API网关配置 | 渊码 |
| 5 | 测试与调优 | 鉴心 |
| 5 | 文档完善 | 小智 |
| 6 | 内部演示 | 灵枢 |

### Phase 2: Beta (6周)

- [ ] 完整ZKP证明系统
- [ ] 远程证明实现
- [ ] 区块链存证
- [ ] 智能合约分账
- [ ] Go/Java SDK
- [ ] 企业级安全加固

### Phase 3: V1 (6周)

- [ ] 多TEE节点支持
- [ ] 联邦学习集成
- [ ] 完整审计追溯
- [ ] 合规报告生成
- [ ] 性能优化

### Phase 4: 生产就绪 (6周)

- [ ] K8s生产部署
- [ ] 高可用架构
- [ ] 灾备方案
- [ ] 安全审计
- [ ] 客户POC

---

## 7. 技术风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| ZKP计算性能 | 高 | GPU加速、批量证明、缓存 |
| TEE侧信道攻击 | 高 | 定期更新微码、防护库 |
| 电路设计错误 | 高 | 形式化验证、审计 |
| 跨语言SDK维护 | 中 | 代码生成、统一规范 |
| SGX硬件依赖 | 中 | 支持SEV、云服务商合作 |

---

**文档版本**: v1.0  
**最后更新**: 2026-04-17  
**作者**: 渊码 (全栈工程师)  
**审核**: 灵枢 (架构师)
