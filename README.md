# vllm-ageserve

资源受限场景下的复合AI系统加速技术研究 —— 基于 vLLM 0.10.2 的二次开发项目。

## 环境部署

### 1. 机器环境

- GPU: 4x NVIDIA RTX 3090 (24GB)
- OS: Ubuntu 20.04
- CUDA Driver: 570.133.07 (支持 CUDA 12.8)

### 2. 安装 CUDA Toolkit

使用 spack 管理 CUDA 版本（已配置为默认加载 12.6.2）：

```bash
source /data/spack/share/spack/setup-env.sh
spack load cuda@12.6.2
```

spack 已写入 `~/.bashrc`，开终端自动加载。

### 3. 创建虚拟环境

```bash
cd vllm-ageserve
uv venv
source .venv/bin/activate
```

### 4. 安装依赖

```bash
# 先装 torch (cu126，与 CUDA 12.6 匹配)
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

# 降级 transformers（清华源默认装的 5.x 不兼容 vLLM 0.10.2）
uv pip install "transformers>=4.55.2,<5.0.0"

# 编译安装 vllm (editable 模式，改代码即生效)
uv pip install -e .
```

第一次编译约需 50 分钟，之后增量修改 Python 代码无需重新编译。

### 5. 配置 git SSH（必需）

机器在隔离网络中，HTTPS 连不上 GitHub，需要用 SSH：

```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

否则编译时 CMake 无法克隆 cutlass、flashmla、vllm-flash-attn 等依赖。

## 开发

```bash
source .venv/bin/activate       # 激活环境
vllm serve <model>              # 启动推理服务
```

改完 Python 代码直接生效，不需要重新编译。

## 启动示例

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 vllm serve /data/models/Qwen3-1.7B \
  --port 8000 \
  --gpu-memory-utilization 0.7
```

测试推理：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```
