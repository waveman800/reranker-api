# 多模态重排服务 - 架构师技术方案

**作者**: 灵枢 (CEO/架构师)  
**日期**: 2026-04-16  
**版本**: v1.0

---

## 1. 整体架构设计

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        客户端层 (Client Layer)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Web App    │  │  Mobile App │  │  CLI Tool   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      接入层 (Gateway Layer)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  API Gateway (Kong/Nginx)                               │    │
│  │  - 负载均衡                                              │    │
│  │  - 速率限制 (Rate Limiting)                              │    │
│  │  - SSL/TLS 终止                                          │    │
│  └────────────────────────┬────────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      安全层 (Security Layer)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  鉴权中心     │  │  反蒸馏保护   │  │  请求审计    │           │
│  │  Auth Center │  │  Anti-Distill│  │  Audit Log   │           │
│  │              │  │              │  │              │           │
│  │ • API Key    │  │ • 输出混淆   │  │ • 请求记录   │           │
│  │ • JWT验证    │  │ • 水印嵌入   │  │ • 行为分析   │           │
│  │ • 签名验证   │  │ • 指纹检测   │  │ • 异常告警   │           │
│  │ • 权限控制   │  │ • 输出加密   │  │ • 合规报告   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      服务层 (Service Layer)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Reranker API Service                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │  Text       │  │  Multimodal │  │  Health     │      │    │
│  │  │  Reranker   │  │  Reranker   │  │  Check      │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  │                                                          │    │
│  │  • FastAPI Framework                                     │    │
│  │  • Async Processing                                      │    │
│  │  • Batch Processing                                      │    │
│  └────────────────────────┬────────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      模型层 (Model Layer)                         │
│  ┌────────────────────────┐  ┌────────────────────────┐         │
│  │  Qwen3-VL-Reranker-2B  │  │  Qwen3-Reranker-4B     │         │
│  │  (Multimodal Model)    │  │  (Text Model)          │         │
│  │                        │  │                        │         │
│  │  • Vision + Text       │  │  • Text Only           │         │
│  │  • GPU Acceleration    │  │  • GPU/CPU Support     │         │
│  └────────────────────────┘  └────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据层 (Data Layer)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Redis       │  │  PostgreSQL  │  │  Object      │           │
│  │  (Cache)     │  │  (Metadata)  │  │  Storage     │           │
│  │              │  │              │  │              │           │
│  │ • Rate Limit │  │ • User Info  │  │ • Model      │           │
│  │ • Session    │  │ • API Keys   │  │   Files      │           │
│  │ • Result     │  │ • Audit Log  │  │ • Uploads    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Ingress Controller                  │   │
│  │              (Nginx / Traefik / Kong)               │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────┼───────────────────────────────┐   │
│  │                     ▼                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   API Pod   │  │   API Pod   │  │   API Pod   │ │   │
│  │  │   (HPA)     │  │   (HPA)     │  │   (HPA)     │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              Model Serving Pod               │   │   │
│  │  │         (GPU Node / Triton Inference)        │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   Redis     │  │  PostgreSQL │  │   MinIO     │ │   │
│  │  │   (HA)      │  │    (HA)     │  │  (Object)   │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 鉴权架构设计

### 2.1 鉴权流程

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────┐
│ Client  │────▶│ API Gateway  │────▶│ Auth Service │────▶│ Backend │
└─────────┘     └──────────────┘     └──────────────┘     └─────────┘
      │                │                    │                  │
      │                │                    │                  │
      │ 1. Request     │                    │                  │
      │    + API Key   │                    │                  │
      │    + Timestamp │                    │                  │
      │    + Signature │                    │                  │
      │───────────────▶│                    │                  │
      │                │                    │                  │
      │                │ 2. Verify Rate     │                  │
      │                │    Limit           │                  │
      │                │────────────────────│                  │
      │                │                    │                  │
      │                │ 3. Validate        │                  │
      │                │    API Key         │                  │
      │                │────────────────────│                  │
      │                │                    │                  │
      │                │ 4. Verify          │                  │
      │                │    Signature       │                  │
      │                │────────────────────│                  │
      │                │                    │                  │
      │                │ 5. Check           │                  │
      │                │    Permissions     │                  │
      │                │────────────────────│                  │
      │                │                    │                  │
      │                │ 6. Forward         │                  │
      │                │    Request         │                  │
      │                │───────────────────────────────────────▶│
      │                │                    │                  │
      │ 7. Response    │                    │                  │
      │◀───────────────│◀───────────────────────────────────────│
```

### 2.2 鉴权方式

#### 2.2.1 API Key 认证
```python
# 请求头
Authorization: Bearer sk-xxxxxxxxxxxxxxxx
X-Request-Timestamp: 1713245678
X-Request-Signature: sha256=abc123...

# 签名算法
signature = HMAC-SHA256(
    key=api_secret,
    message=f"{timestamp}:{method}:{path}:{body_hash}"
)
```

#### 2.2.2 JWT Token 认证
```python
# 请求头
Authorization: Bearer eyJhbGciOiJSUzI1Ni...

# JWT Payload
{
    "sub": "user_id",
    "iss": "auth.service",
    "iat": 1713245678,
    "exp": 1713249278,
    "scope": "rerank:read rerank:write",
    "quota": {
        "requests_per_minute": 100,
        "requests_per_day": 10000
    }
}
```

#### 2.2.3 请求签名验证
```python
class RequestSignature:
    """请求签名验证"""
    
    def verify(self, request):
        # 1. 提取签名参数
        timestamp = request.headers.get('X-Request-Timestamp')
        signature = request.headers.get('X-Request-Signature')
        
        # 2. 验证时间戳（防重放）
        if abs(time.time() - int(timestamp)) > 300:  # 5分钟窗口
            raise AuthenticationError("Request expired")
        
        # 3. 验证签名
        expected = self._generate_signature(request)
        if not hmac.compare_digest(signature, expected):
            raise AuthenticationError("Invalid signature")
    
    def _generate_signature(self, request):
        # 构造签名内容
        content = f"{request.timestamp}:{request.method}:{request.path}:{request.body_hash}"
        return hmac.new(self.secret_key, content.encode(), hashlib.sha256).hexdigest()
```

### 2.3 权限控制模型

```python
# RBAC + ABAC 混合模型
class Permission:
    """权限控制"""
    
    ROLES = {
        "admin": ["*"],  # 所有权限
        "developer": [
            "rerank:read",
            "rerank:write",
            "model:read",
            "stats:read"
        ],
        "readonly": [
            "rerank:read",
            "model:read"
        ]
    }
    
    QUOTAS = {
        "free": {
            "rpm": 10,      # requests per minute
            "rpd": 1000,    # requests per day
            "batch_max": 10
        },
        "pro": {
            "rpm": 100,
            "rpd": 100000,
            "batch_max": 100
        },
        "enterprise": {
            "rpm": 1000,
            "rpd": 1000000,
            "batch_max": 1000
        }
    }
```

---

## 3. 反蒸馏架构设计

### 3.1 反蒸馏策略

```
┌─────────────────────────────────────────────────────────────┐
│                    反蒸馏保护系统                             │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   输入层     │  │   处理层     │  │   输出层     │      │
│  │  Protection  │  │  Protection  │  │  Protection  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 请求指纹     │  │ 输出混淆     │  │ 水印嵌入     │      │
│  │ 识别         │  │              │  │              │      │
│  │              │  │              │  │              │      │
│  │ • IP分析     │  │ • 分数扰动   │  │ • 隐形水印   │      │
│  │ • 行为模式   │  │ • 随机噪声   │  │ • 语义水印   │      │
│  │ • 频率检测   │  │ • 结果截断   │  │ • 溯源标识   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              异常行为检测引擎                         │   │
│  │                                                     │   │
│  │  • 批量请求异常检测                                  │   │
│  │  • 查询模式分析（提取训练数据？）                     │   │
│  │  • 输出分布异常检测                                  │   │
│  │  • 实时风险评分                                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 输出混淆技术

```python
class OutputObfuscation:
    """输出混淆保护"""
    
    def __init__(self, noise_level=0.01, seed=None):
        self.noise_level = noise_level
        self.seed = seed
    
    def obfuscate_scores(self, scores: List[float]) -> List[float]:
        """分数混淆"""
        # 1. 添加可控噪声
        noise = self._generate_noise(len(scores))
        obfuscated = [s + n for s, n in zip(scores, noise)]
        
        # 2. 保持相对顺序
        obfuscated = self._preserve_ranking(obfuscated)
        
        # 3. 截断精度（防止精确反推）
        obfuscated = [round(s, 4) for s in obfuscated]
        
        return obfuscated
    
    def _generate_noise(self, n: int) -> List[float]:
        """生成可控噪声"""
        rng = random.Random(self.seed)
        return [rng.gauss(0, self.noise_level) for _ in range(n)]
    
    def _preserve_ranking(self, scores: List[float]) -> List[float]:
        """保持排序一致性"""
        # 确保排序不变但数值有扰动
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        
        # 添加微小差异保持顺序
        for i, (idx, _) in enumerate(indexed):
            scores[idx] += i * 1e-7
        
        return scores
```

### 3.3 水印嵌入技术

```python
class SemanticWatermark:
    """语义水印嵌入"""
    
    def __init__(self, watermark_key: str):
        self.key = watermark_key
        self.hash_func = hashlib.blake2b
    
    def embed(self, text: str, user_id: str) -> str:
        """在文本中嵌入水印"""
        # 1. 生成用户特定水印
        watermark = self._generate_watermark(user_id)
        
        # 2. 选择嵌入位置（基于文本特征）
        positions = self._select_positions(text, watermark)
        
        # 3. 嵌入水印（同义词替换、句式微调）
        watermarked = self._embed_at_positions(text, positions, watermark)
        
        return watermarked
    
    def extract(self, text: str) -> Optional[str]:
        """提取水印信息"""
        # 分析文本特征，尝试提取水印
        # 返回用户ID或None
        pass
    
    def _generate_watermark(self, user_id: str) -> str:
        """生成用户特定水印"""
        return self.hash_func(
            f"{self.key}:{user_id}".encode()
        ).hexdigest()[:16]
```

### 3.4 请求指纹识别

```python
class RequestFingerprint:
    """请求指纹识别"""
    
    def analyze(self, request) -> Dict:
        """分析请求特征"""
        fingerprint = {
            # 网络特征
            "ip": self._hash_ip(request.client_ip),
            "user_agent_hash": hashlib.md5(
                request.headers.get('User-Agent', '').encode()
            ).hexdigest()[:8],
            
            # 行为特征
            "query_pattern": self._extract_pattern(request.query),
            "request_frequency": self._get_frequency(request.api_key),
            "time_distribution": self._get_time_distribution(request.api_key),
            
            # 内容特征
            "query_complexity": self._analyze_complexity(request.query),
            "document_types": self._analyze_doc_types(request.documents),
        }
        
        return fingerprint
    
    def detect_anomaly(self, fingerprint: Dict) -> RiskScore:
        """检测异常行为"""
        risk = RiskScore()
        
        # 1. 高频请求检测
        if fingerprint["request_frequency"] > THRESHOLD:
            risk.add("high_frequency", 0.3)
        
        # 2. 查询模式异常
        if self._is_training_pattern(fingerprint["query_pattern"]):
            risk.add("training_pattern", 0.5)
        
        # 3. 时间分布异常（自动化工具？）
        if fingerprint["time_distribution"] < 0.1:  # 过于均匀
            risk.add("automated_pattern", 0.2)
        
        return risk
```

---

## 4. 性能优化设计

### 4.1 缓存策略

```python
class CacheStrategy:
    """多级缓存策略"""
    
    def __init__(self):
        self.local_cache = {}  # L1: 本地内存
        self.redis_cache = None  # L2: Redis
        self.db_cache = None  # L3: DB
    
    def get(self, key: str) -> Optional[Any]:
        # L1
        if key in self.local_cache:
            return self.local_cache[key]
        
        # L2
        value = self.redis_cache.get(key)
        if value:
            self.local_cache[key] = value
            return value
        
        # L3
        value = self.db_cache.get(key)
        if value:
            self.redis_cache.set(key, value)
            self.local_cache[key] = value
        
        return value
```

### 4.2 批处理优化

```python
class BatchProcessor:
    """批处理优化"""
    
    def __init__(self, batch_size: int = 32, max_wait: float = 0.1):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.queue = asyncio.Queue()
    
    async def submit(self, request) -> Future:
        """提交请求到批处理队列"""
        future = asyncio.Future()
        await self.queue.put((request, future))
        return future
    
    async def process_loop(self):
        """批处理循环"""
        while True:
            batch = []
            deadline = asyncio.get_event_loop().time() + self.max_wait
            
            # 收集批次
            while len(batch) < self.batch_size:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=max(0, timeout)
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            # 处理批次
            if batch:
                await self._process_batch(batch)
```

---

## 5. 监控与告警

### 5.1 监控指标

```python
METRICS = {
    # 性能指标
    "request_latency": "Histogram",
    "request_rate": "Counter",
    "batch_size": "Gauge",
    "queue_length": "Gauge",
    
    # 业务指标
    "rerank_requests_total": "Counter",
    "rerank_errors_total": "Counter",
    "cache_hit_rate": "Gauge",
    
    # 安全指标
    "auth_failures_total": "Counter",
    "rate_limit_hits_total": "Counter",
    "suspicious_requests_total": "Counter",
    "watermark_detections_total": "Counter",
}
```

### 5.2 告警规则

```yaml
alerts:
  - name: HighLatency
    condition: latency_p99 > 5000  # 5s
    severity: warning
    
  - name: HighErrorRate
    condition: error_rate > 0.01  # 1%
    severity: critical
    
  - name: SuspiciousActivity
    condition: suspicious_requests > 100
    severity: warning
    
  - name: AuthAttack
    condition: auth_failures > 1000
    severity: critical
```

---

## 6. 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| API框架 | FastAPI | 高性能、异步支持、自动生成文档 |
| 网关 | Kong/Nginx | 成熟稳定、插件丰富 |
| 缓存 | Redis | 高性能、支持分布式 |
| 数据库 | PostgreSQL | 可靠、支持JSON |
| 对象存储 | MinIO | 兼容S3、轻量 |
| 监控 | Prometheus + Grafana | 云原生标准 |
| 日志 | ELK Stack | 全文检索、可视化 |
| 部署 | Kubernetes | 弹性伸缩、高可用 |

---

## 7. 安全加固

### 7.1 网络安全
- TLS 1.3 加密传输
- mTLS 服务间通信
- WAF 防护
- DDoS 防护

### 7.2 应用安全
- 输入验证和清洗
- SQL注入防护
- XSS防护
- CSRF防护

### 7.3 数据安全
- 敏感数据加密存储
- 密钥轮换机制
- 数据脱敏
- 审计日志

---

## 8. 实施路线图

### Phase 1: 基础功能 (2周)
- [ ] 多模态重排API
- [ ] 基础鉴权（API Key）
- [ ] 速率限制
- [ ] Docker部署

### Phase 2: 安全加固 (2周)
- [ ] JWT认证
- [ ] 请求签名
- [ ] 输出混淆
- [ ] 基础监控

### Phase 3: 高级保护 (2周)
- [ ] 水印嵌入
- [ ] 请求指纹识别
- [ ] 异常行为检测
- [ ] 完整审计

### Phase 4: 生产就绪 (2周)
- [ ] K8s部署
- [ ] 性能优化
- [ ] 高可用架构
- [ ] 灾备方案

---

**文档版本**: v1.0  
**最后更新**: 2026-04-16  
**作者**: 灵枢 (CEO/架构师)
