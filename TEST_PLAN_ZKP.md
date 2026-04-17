# 可信数据护盾 - ZKP测试计划

**文档名称**: 可信数据护盾ZKP测试计划  
**版本**: v1.0.0  
**创建日期**: 2026-04-17  
**创建人**: 鉴心 (测试专家)  
**关联项目**: Trusted Data Shield (可信数据护盾)  
**关联文档**: 
- PRODUCT_DEFINITIVE.md (产品定义)
- TECH_IMPLEMENTATION.md (技术实现)
- PRD.md (产品需求)

---

## 目录

1. [测试概述](#1-测试概述)
2. [ZKP专项测试](#2-zkp专项测试)
3. [TEE安全测试](#3-tee安全测试)
4. [反蒸馏测试](#4-反蒸馏测试)
5. [集成测试](#5-集成测试)
6. [性能测试](#6-性能测试)
7. [安全测试](#7-安全测试)
8. [测试环境](#8-测试环境)
9. [测试工具](#9-测试工具)
10. [测试排期](#10-测试排期)

---

## 1. 测试概述

### 1.1 项目背景

可信数据护盾是基于**零知识证明（ZKP）**、**可信执行环境（TEE）**和**区块链**技术的ToB数据服务平台。核心能力包括：
- **ZKP数据质量证明**: 证明数据质量而不泄露原始数据
- **TEE安全计算**: 在可信环境内执行数据处理和模型推理
- **反蒸馏保护**: 防止模型记忆和复制原始数据

### 1.2 测试目标

| 目标 | 描述 | 验收标准 |
|------|------|----------|
| ZKP正确性 | 证明生成和验证逻辑正确 | 100%测试通过 |
| ZKP安全性 | 零知识属性、不可伪造性 | 通过形式化验证 |
| TEE隔离性 | 内存隔离、侧信道防护 | 通过SGX安全测试 |
| 反蒸馏有效性 | 水印可检测、行为可追溯 | 检测率>95% |
| 系统稳定性 | 7x24小时稳定运行 | MTBF>720小时 |

### 1.3 测试范围

**包含**:
- ZKP电路功能测试
- ZKP证明生成/验证测试
- TEE环境测试
- 远程证明测试
- 反蒸馏保护测试
- 区块链存证测试
- 端到端集成测试

**不包含**:
- 底层密码学算法测试（依赖库）
- SGX硬件故障测试
- 区块链共识测试

---

## 2. ZKP专项测试

### 2.1 电路功能测试

#### 2.1.1 数据质量证明电路

```python
# test_zkp_circuit.py

import pytest
from circom_tester import wasm_tester

class TestDataQualityCircuit:
    """数据质量证明电路测试"""
    
    def test_completeness_calculation(self):
        """测试完整性计算逻辑"""
        circuit = wasm_tester("circuits/data_quality.wasm")
        
        # 完整数据
        input_data = {
            "data": [1.0] * 100,  # 100条完整数据
            "labels": [0] * 50 + [1] * 50,  # 平衡标签
            "data_hash": "0x...",
            "quality_score": 0.95,
            "label_distribution": [0.5, 0.5] + [0] * 8
        }
        
        witness = circuit.calculate_witness(input_data)
        assert witness["completeness"] == 1.0
        
    def test_balance_calculation(self):
        """测试标签平衡性计算"""
        circuit = wasm_tester("circuits/data_quality.wasm")
        
        # 不平衡数据
        input_data = {
            "data": [1.0] * 100,
            "labels": [0] * 90 + [1] * 10,  # 严重不平衡
            "data_hash": "0x...",
            "quality_score": 0.55,
            "label_distribution": [0.9, 0.1] + [0] * 8
        }
        
        witness = circuit.calculate_witness(input_data)
        assert witness["balance"] < 0.5
        
    def test_hash_constraint(self):
        """测试哈希约束"""
        circuit = wasm_tester("circuits/data_quality.wasm")
        
        # 错误哈希
        input_data = {
            "data": [1.0] * 100,
            "labels": [0] * 50 + [1] * 50,
            "data_hash": "0xWRONG",  # 错误哈希
            "quality_score": 0.95,
            "label_distribution": [0.5, 0.5] + [0] * 8
        }
        
        with pytest.raises(ConstraintSatisfactionError):
            circuit.calculate_witness(input_data)
    
    def test_score_range_constraint(self):
        """测试分数范围约束"""
        circuit = wasm_tester("circuits/data_quality.wasm")
        
        # 超出范围的分数
        input_data = {
            "data": [1.0] * 100,
            "labels": [0] * 50 + [1] * 50,
            "data_hash": "0x...",
            "quality_score": 1.5,  # 超出范围
            "label_distribution": [0.5, 0.5] + [0] * 8
        }
        
        with pytest.raises(ConstraintSatisfactionError):
            circuit.calculate_witness(input_data)
```

#### 2.1.2 计算完整性证明电路

```python
# test_computation_circuit.py

class TestComputationIntegrityCircuit:
    """计算完整性证明电路测试"""
    
    def test_reranker_computation(self):
        """测试重排序计算完整性"""
        circuit = wasm_tester("circuits/computation_integrity.wasm")
        
        input_data = {
            "query_embedding": [0.1] * 768,
            "doc_embeddings": [[0.2] * 768] * 10,
            "expected_scores": [0.5] * 10,
            "computation_hash": "0x..."
        }
        
        witness = circuit.calculate_witness(input_data)
        assert witness["computation_valid"] == 1
        
    def test_wrong_computation_detection(self):
        """测试错误计算检测"""
        circuit = wasm_tester("circuits/computation_integrity.wasm")
        
        # 错误的计算结果
        input_data = {
            "query_embedding": [0.1] * 768,
            "doc_embeddings": [[0.2] * 768] * 10,
            "expected_scores": [0.9] * 10,  # 错误分数
            "computation_hash": "0x..."
        }
        
        with pytest.raises(ConstraintSatisfactionError):
            circuit.calculate_witness(input_data)
```

### 2.2 证明生成测试

```python
# test_proof_generation.py

import pytest
from zkp_service import ZKPService

class TestProofGeneration:
    """ZKP证明生成测试"""
    
    @pytest.fixture
    def zkp_service(self):
        return ZKPService(params_path="/params/data_quality.json")
    
    def test_successful_proof_generation(self, zkp_service):
        """测试成功生成证明"""
        data = generate_test_data(n=100, quality="high")
        labels = generate_balanced_labels(n=100, num_classes=10)
        
        proof = zkp_service.prove_data_quality(data, labels)
        
        assert proof is not None
        assert proof.proof_bytes is not None
        assert len(proof.public_inputs) > 0
        assert proof.data_hash is not None
        assert 0 <= proof.quality_score <= 1
        
    def test_proof_generation_performance(self, zkp_service):
        """测试证明生成性能"""
        import time
        
        data_sizes = [10, 100, 1000]
        max_times = [1, 5, 30]  # 秒
        
        for size, max_time in zip(data_sizes, max_times):
            data = generate_test_data(n=size)
            labels = generate_balanced_labels(n=size)
            
            start = time.time()
            proof = zkp_service.prove_data_quality(data, labels)
            elapsed = time.time() - start
            
            assert elapsed < max_time, f"Size {size} took {elapsed}s > {max_time}s"
            
    def test_proof_determinism(self, zkp_service):
        """测试证明确定性（相同输入产生不同但等效证明）"""
        data = generate_test_data(n=100)
        labels = generate_balanced_labels(n=100)
        
        proof1 = zkp_service.prove_data_quality(data, labels)
        proof2 = zkp_service.prove_data_quality(data, labels)
        
        # 证明字节不同（随机性）
        assert proof1.proof_bytes != proof2.proof_bytes
        # 但公开输入相同
        assert proof1.public_inputs == proof2.public_inputs
        # 都验证通过
        assert zkp_service.verify(proof1)
        assert zkp_service.verify(proof2)
```

### 2.3 证明验证测试

```python
# test_proof_verification.py

class TestProofVerification:
    """ZKP证明验证测试"""
    
    def test_valid_proof_verification(self, zkp_service):
        """测试有效证明验证"""
        data = generate_test_data(n=100)
        labels = generate_balanced_labels(n=100)
        
        proof = zkp_service.prove_data_quality(data, labels)
        is_valid = zkp_service.verify(proof)
        
        assert is_valid is True
        
    def test_invalid_proof_rejection(self, zkp_service):
        """测试无效证明拒绝"""
        data = generate_test_data(n=100)
        labels = generate_balanced_labels(n=100)
        
        proof = zkp_service.prove_data_quality(data, labels)
        
        # 篡改证明
        proof.proof_bytes = proof.proof_bytes[:-10] + b'\x00' * 10
        
        is_valid = zkp_service.verify(proof)
        assert is_valid is False
        
    def test_wrong_public_inputs(self, zkp_service):
        """测试错误公开输入拒绝"""
        data = generate_test_data(n=100)
        labels = generate_balanced_labels(n=100)
        
        proof = zkp_service.prove_data_quality(data, labels)
        
        # 篡改公开输入
        proof.public_inputs[0] = "0xWRONG"
        
        is_valid = zkp_service.verify(proof)
        assert is_valid is False
        
    def test_batch_verification(self, zkp_service):
        """测试批量验证"""
        proofs = []
        for _ in range(100):
            data = generate_test_data(n=100)
            labels = generate_balanced_labels(n=100)
            proofs.append(zkp_service.prove_data_quality(data, labels))
        
        # 批量验证
        results = zkp_service.verify_batch(proofs)
        assert all(results)
        
    def test_verification_performance(self, zkp_service):
        """测试验证性能"""
        import time
        
        data = generate_test_data(n=100)
        labels = generate_balanced_labels(n=100)
        proof = zkp_service.prove_data_quality(data, labels)
        
        start = time.time()
        for _ in range(1000):
            zkp_service.verify(proof)
        elapsed = time.time() - start
        
        # 1000次验证应<10秒
        assert elapsed < 10, f"Verification too slow: {elapsed}s"
```

### 2.4 零知识属性测试

```python
# test_zero_knowledge.py

class TestZeroKnowledgeProperty:
    """零知识属性测试"""
    
    def test_proof_reveals_no_private_data(self, zkp_service):
        """测试证明不泄露私有数据"""
        data = ["sensitive_info_" + str(i) for i in range(100)]
        labels = [0] * 50 + [1] * 50
        
        proof = zkp_service.prove_data_quality(data, labels)
        
        # 证明中不应包含原始数据
        proof_str = str(proof.proof_bytes)
        for item in data:
            assert item not in proof_str
            
    def test_simulation_indistinguishability(self, zkp_service):
        """测试模拟器不可区分性"""
        from zkp_simulator import simulate_proof
        
        # 生成真实证明
        data = generate_test_data(n=100)
        labels = generate_balanced_labels(n=100)
        real_proof = zkp_service.prove_data_quality(data, labels)
        
        # 模拟证明（不知道私有输入）
        simulated_proof = simulate_proof(real_proof.public_inputs)
        
        # 两者应不可区分（统计测试）
        assert statistical_indistinguishable(real_proof, simulated_proof)
```

---

## 3. TEE安全测试

### 3.1 内存隔离测试

```python
# test_tee_isolation.py

import pytest
import subprocess

class TestTEEIsolation:
    """TEE内存隔离测试"""
    
    def test_enclave_memory_protection(self):
        """测试Enclave内存保护"""
        # 尝试从外部访问Enclave内存
        result = subprocess.run(
            ["./attack_tools/read_enclave_memory"],
            capture_output=True
        )
        
        # 应失败
        assert result.returncode != 0
        assert b"Permission denied" in result.stderr or result.returncode == -11
        
    def test_data_sealing(self):
        """测试数据密封"""
        from tee_client import TEEClient
        
        client = TEEClient()
        sensitive_data = b"top_secret_data"
        
        # 密封数据
        sealed = client.seal_data(sensitive_data)
        
        # 解封数据
        unsealed = client.unseal_data(sealed)
        
        assert unsealed == sensitive_data
        
        # 尝试用不同Enclave解封
        result = client.unseal_with_different_enclave(sealed)
        assert result is None or result != sensitive_data
```

### 3.2 远程证明测试

```python
# test_remote_attestation.py

class TestRemoteAttestation:
    """远程证明测试"""
    
    def test_quote_generation(self):
        """测试Quote生成"""
        from sgx_attestation import generate_quote
        
        quote = generate_quote()
        
        assert quote is not None
        assert len(quote) > 0
        assert quote.version == 3  # SGX Quote version
        
    def test_quote_verification(self):
        """测试Quote验证"""
        from sgx_attestation import generate_quote, verify_quote
        
        quote = generate_quote()
        
        # 使用Intel PCS验证
        result = verify_quote(quote, ias_api_key="test_key")
        
        assert result.is_valid is True
        assert result.enclave_identity.measurement is not None
        
    def test_enclave_identity_binding(self):
        """测试Enclave身份绑定"""
        from tee_client import TEEClient
        
        client = TEEClient()
        
        # 获取Enclave身份
        identity = client.get_enclave_identity()
        
        # 验证身份与代码度量匹配
        expected_measurement = calculate_expected_measurement()
        assert identity.mr_enclave == expected_measurement
```

### 3.3 侧信道防护测试

```python
# test_side_channel.py

class TestSideChannelProtection:
    """侧信道攻击防护测试"""
    
    def test_constant_time_operations(self):
        """测试常量时间操作"""
        import time
        
        # 测试不同输入的执行时间
        times = []
        for secret in [b'\x00' * 32, b'\xff' * 32, b'\xab' * 32]:
            start = time.perf_counter_ns()
            tee_hmac(secret, b'message')
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)
        
        # 时间差异应<5%
        max_time = max(times)
        min_time = min(times)
        assert (max_time - min_time) / max_time < 0.05
        
    def test_cache_attack_resistance(self):
        """测试缓存攻击抵抗"""
        # Flush+Reload攻击测试
        result = run_cache_attack_simulation()
        
        # 不应泄露信息
        assert result.information_leaked is False
```

---

## 4. 反蒸馏测试

### 4.1 水印测试

```python
# test_watermark.py

class TestWatermark:
    """水印嵌入与检测测试"""
    
    def test_watermark_embedding(self):
        """测试水印嵌入"""
        from anti_distillation import WatermarkEngine
        
        engine = WatermarkEngine(key="secret_key")
        data = [0.5] * 100
        user_id = "user_123"
        
        watermarked = engine.embed(data, user_id)
        
        # 数据应被修改但保持排序
        assert watermarked != data
        assert sorted(range(len(watermarked)), key=lambda i: watermarked[i], reverse=True) == \
               sorted(range(len(data)), key=lambda i: data[i], reverse=True)
        
    def test_watermark_detection(self):
        """测试水印检测"""
        from anti_distillation import WatermarkEngine
        
        engine = WatermarkEngine(key="secret_key")
        data = [0.5] * 100
        user_id = "user_123"
        
        watermarked = engine.embed(data, user_id)
        
        # 检测水印
        detected_user = engine.detect(watermarked)
        
        assert detected_user == user_id
        
    def test_watermark_robustness(self):
        """测试水印鲁棒性"""
        from anti_distillation import WatermarkEngine
        
        engine = WatermarkEngine(key="secret_key")
        data = [0.5] * 100
        user_id = "user_123"
        
        watermarked = engine.embed(data, user_id)
        
        # 模拟攻击：添加噪声
        attacked = [w + np.random.normal(0, 0.001) for w in watermarked]
        
        # 仍应能检测
        detected_user = engine.detect(attacked)
        assert detected_user == user_id
```

### 4.2 行为监控测试

```python
# test_behavior_monitoring.py

class TestBehaviorMonitoring:
    """行为监控测试"""
    
    def test_query_pattern_detection(self):
        """测试查询模式检测"""
        from behavior_monitor import BehaviorMonitor
        
        monitor = BehaviorMonitor()
        user_id = "user_123"
        
        # 模拟正常查询
        for i in range(100):
            monitor.log_query(user_id, f"query_{i % 10}", [0.1] * 10)
        
        # 应无异常
        assert monitor.check_anomaly(user_id) is False
        
        # 模拟异常查询（全覆盖）
        for i in range(1000):
            monitor.log_query(user_id, f"query_{i}", [0.1] * 10)
        
        # 应检测到异常
        assert monitor.check_anomaly(user_id) is True
        
    def test_rate_limiting(self):
        """测试速率限制"""
        from behavior_monitor import RateLimiter
        
        limiter = RateLimiter(max_requests=10, window=60)
        user_id = "user_123"
        
        # 10次应通过
        for _ in range(10):
            assert limiter.allow(user_id) is True
        
        # 第11次应拒绝
        assert limiter.allow(user_id) is False
```

---

## 5. 集成测试

### 5.1 端到端流程测试

```python
# test_end_to_end.py

class TestEndToEnd:
    """端到端集成测试"""
    
    def test_full_data_sharing_flow(self):
        """测试完整数据共享流程"""
        # 1. 数据提供方上传数据
        provider = DataProvider()
        data_id = provider.upload_data(
            data=generate_test_dataset(),
            metadata={"industry": "finance", "size": 10000}
        )
        
        # 2. 生成ZKP证明
        proof = provider.generate_zkp_proof(data_id)
        assert proof is not None
        
        # 3. 大模型厂商验证证明
        consumer = ModelConsumer()
        is_valid = consumer.verify_zkp_proof(proof)
        assert is_valid is True
        
        # 4. 在TEE内使用数据
        result = consumer.request_rerank(data_id, "test query")
        assert result.scores is not None
        
        # 5. 验证反蒸馏保护
        assert result.watermark is not None
        
        # 6. 审计记录
        audit_record = get_audit_record(data_id)
        assert audit_record.data_access_count == 1
        
    def test_multi_party_computation(self):
        """测试多方计算场景"""
        # 多个数据提供方
        providers = [DataProvider() for _ in range(3)]
        data_ids = []
        
        for provider in providers:
            data_id = provider.upload_data(generate_test_dataset())
            data_ids.append(data_id)
        
        # 大模型厂商聚合查询
        consumer = ModelConsumer()
        results = consumer.batch_rerank(data_ids, "query")
        
        assert len(results) == 3
        for result in results:
            assert result.watermark is not None
```

### 5.2 故障恢复测试

```python
# test_fault_tolerance.py

class TestFaultTolerance:
    """故障恢复测试"""
    
    def test_zkp_service_restart(self):
        """测试ZKP服务重启"""
        # 生成证明
        proof = generate_proof()
        
        # 重启服务
        restart_zkp_service()
        
        # 验证证明仍有效
        assert verify_proof(proof) is True
        
    def test_tee_enclave_restart(self):
        """测试TEE Enclave重启"""
        # 密封数据
        sealed = seal_data(b"secret")
        
        # 重启Enclave
        restart_enclave()
        
        # 解封数据
        unsealed = unseal_data(sealed)
        assert unsealed == b"secret"
```

---

## 6. 性能测试

### 6.1 ZKP性能测试

```python
# test_zkp_performance.py

class TestZKPPerformance:
    """ZKP性能测试"""
    
    def test_proof_generation_throughput(self):
        """测试证明生成吞吐量"""
        import concurrent.futures
        import time
        
        def generate_one():
            data = generate_test_data(n=100)
            labels = generate_balanced_labels(n=100)
            return zkp_service.prove_data_quality(data, labels)
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(generate_one) for _ in range(100)]
            proofs = [f.result() for f in futures]
        elapsed = time.time() - start
        
        throughput = 100 / elapsed
        print(f"Throughput: {throughput} proofs/sec")
        assert throughput > 5  # 至少5个/秒
        
    def test_proof_size(self):
        """测试证明大小"""
        data = generate_test_data(n=100)
        labels = generate_balanced_labels(n=100)
        proof = zkp_service.prove_data_quality(data, labels)
        
        proof_size = len(proof.proof_bytes)
        print(f"Proof size: {proof_size} bytes")
        
        # Groth16证明应<500字节
        assert proof_size < 500
```

### 6.2 TEE性能测试

```python
# test_tee_performance.py

class TestTEEPerformance:
    """TEE性能测试"""
    
    def test_enclave_switch_overhead(self):
        """测试Enclave切换开销"""
        import time
        
        times = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            tee_ecall("nop")  # 空操作
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"ECall overhead: {avg_time / 1000} us")
        
        # ECall开销应<100us
        assert avg_time < 100_000
        
    def test_memory_bandwidth(self):
        """测试TEE内存带宽"""
        import time
        
        data_size = 100 * 1024 * 1024  # 100MB
        
        start = time.time()
        tee_copy_in(b'\x00' * data_size)
        elapsed = time.time() - start
        
        bandwidth = data_size / elapsed / 1024 / 1024
        print(f"Memory bandwidth: {bandwidth} MB/s")
        
        # 带宽应>100MB/s
        assert bandwidth > 100
```

---

## 7. 安全测试

### 7.1 渗透测试

```python
# test_penetration.py

class TestPenetration:
    """渗透测试"""
    
    def test_api_authentication_bypass(self):
        """测试API认证绕过"""
        import requests
        
        # 无认证请求
        response = requests.post(
            "http://api.zkp-service/v1/prove",
            json={"data": "test"}
        )
        
        assert response.status_code == 401
        
    def test_proof_replay_attack(self):
        """测试证明重放攻击"""
        # 生成有效证明
        proof = generate_valid_proof()
        
        # 首次验证
        assert verify_proof(proof) is True
        
        # 重放（应失败，因为nonce已使用）
        result = verify_proof(proof)
        assert result is False or result.error == "nonce_reused"
        
    def test_circuit_underconstrained(self):
        """测试电路欠约束"""
        # 尝试找到满足约束但逻辑错误的输入
        from zkp_fuzzer import fuzz_circuit
        
        issues = fuzz_circuit("circuits/data_quality.circom", iterations=10000)
        
        assert len(issues) == 0, f"Found underconstrained issues: {issues}"
```

### 7.2 形式化验证

```python
# test_formal_verification.py

class TestFormalVerification:
    """形式化验证"""
    
    def test_circuit_soundness(self):
        """测试电路可靠性"""
        # 使用circomspect分析
        result = run_circomspect("circuits/data_quality.circom")
        
        assert result.vulnerabilities == 0
        
    def test_zero_knowledge_property(self):
        """测试零知识属性"""
        # 使用zkalc验证
        from zkalc_verifier import verify_zk_property
        
        result = verify_zk_property(
            circuit="circuits/data_quality.circom",
            proof_system="groth16"
        )
        
        assert result.zero_knowledge is True
        assert result.soundness is True
        assert result.completeness is True
```

---

## 8. 测试环境

### 8.1 硬件环境

| 组件 | 配置 | 数量 |
|------|------|------|
| SGX服务器 | Intel Xeon with SGX2, 64GB RAM | 2台 |
| GPU服务器 | NVIDIA A100, 128GB RAM | 1台 |
| 测试客户端 | Standard VM, 16GB RAM | 5台 |

### 8.2 软件环境

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  zkp-test:
    image: trusted-data-shield/zkp:test
    environment:
      - TEST_MODE=true
      - CIRCUIT_PATH=/circuits
    volumes:
      - ./circuits:/circuits:ro
      - ./test-results:/results
    
  tee-test:
    image: trusted-data-shield/tee:test
    devices:
      - /dev/sgx_enclave
    environment:
      - SGX_MODE=HW
      - TEST_MODE=true
    
  blockchain-test:
    image: hyperledger/fabric-peer:2.4
    environment:
      - CORE_PEER_ID=peer0.test
```

---

## 9. 测试工具

| 工具 | 用途 | 版本 |
|------|------|------|
| circom_tester | 电路测试 | ^0.1.0 |
| snarkjs | ZKP操作 | ^0.7.0 |
| pytest | 测试框架 | ^7.0 |
| k6 | 性能测试 | ^0.44 |
| OWASP ZAP | 渗透测试 | ^2.12 |
| circomspect | 电路分析 | ^0.5.0 |
| Intel SGX SDK | TEE开发 | 2.19 |

---

## 10. 测试排期

### Phase 1: 单元测试 (Week 1-2)

| 周 | 任务 | 负责人 |
|----|------|--------|
| 1 | ZKP电路单元测试 | 鉴心 |
| 1 | TEE基础功能测试 | 鉴心 |
| 2 | 反蒸馏单元测试 | 鉴心 |
| 2 | SDK单元测试 | 鉴心 |

### Phase 2: 集成测试 (Week 3-4)

| 周 | 任务 | 负责人 |
|----|------|--------|
| 3 | ZKP-TEE集成测试 | 鉴心 |
| 3 | 端到端流程测试 | 鉴心 |
| 4 | 性能基准测试 | 鉴心 |
| 4 | 安全渗透测试 | 鉴心 |

### Phase 3: 验收测试 (Week 5-6)

| 周 | 任务 | 负责人 |
|----|------|--------|
| 5 | 形式化验证 | 鉴心 |
| 5 | 压力测试 | 鉴心 |
| 6 | 用户验收测试 | 鉴心 |
| 6 | 测试报告 | 鉴心 |

---

## 附录

### A. 测试通过标准

- **P0测试**: 100%通过
- **P1测试**: >95%通过
- **性能测试**: 达到设计指标的90%
- **安全测试**: 无高危漏洞

### B. 缺陷分级

| 级别 | 定义 | 修复时限 |
|------|------|----------|
| P0 | 系统不可用/数据泄露 | 24小时 |
| P1 | 主要功能失效 | 3天 |
| P2 | 次要功能问题 | 1周 |
| P3 | 用户体验问题 | 2周 |

---

**文档版本**: v1.0  
**最后更新**: 2026-04-17  
**作者**: 鉴心 (测试专家)  
**审核**: 灵枢 (架构师)
