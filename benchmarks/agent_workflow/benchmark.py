#!/usr/bin/env python3
"""
Agent Workflow Benchmark — 单模型多 Agent 链式工作流 KV Cache 性能测试。

场景: Code Analyst → Code Fixer → Tester → Reviewer 循环 N 轮。
每个 Agent 有固定的 system prompt (应被 KV cache 命中) + 上一轮 Agent 的输出 (动态)。

用法:
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=1 python benchmarks/agent_workflow/benchmark.py \
      --model /data/models/Qwen3-1.7B --rounds 10

输出: 每轮端到端延迟、KV cache 命中率、平均统计。
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import requests


# ============================================================
# Agent 定义
# ============================================================

AGENT_SYSTEM_PROMPTS = {
    "CodeAnalyst": (
        "You are a senior code analyst. Your job is to carefully review the given code "
        "and identify all potential issues, including bugs, performance problems, security "
        "vulnerabilities, and code style violations. For each issue, provide the file name, "
        "line number, severity level (critical/high/medium/low), and a brief description. "
        "Be thorough and systematic in your analysis. Do NOT suggest fixes — only identify "
        "and describe the problems. Format your output as a numbered list of findings."
    ),
    "CodeFixer": (
        "You are an expert software engineer specializing in bug fixing. You will receive "
        "a code analysis report listing identified issues. Your task is to produce concrete "
        "code patches for each issue. For each fix, show the original code snippet, the "
        "modified code snippet, and explain why the fix resolves the issue. Ensure your "
        "fixes do not introduce new bugs or regressions. Be precise about line numbers "
        "and file paths. Format each fix clearly with '---BEFORE---' and '---AFTER---' blocks."
    ),
    "Tester": (
        "You are a QA test engineer. You will receive a list of code fixes that have been "
        "applied. Your job is to design and write test cases that verify each fix works "
        "correctly and that no regressions have been introduced. For each fix, write at "
        "least one unit test and describe any integration or edge case tests needed. "
        "Specify the expected pass/fail behavior for each test. Format your output with "
        "clear test names, steps, and expected results."
    ),
    "Reviewer": (
        "You are a principal engineer conducting a final code review. You will receive "
        "the original analysis, the applied fixes, and the test plan. Your job is to do "
        "a holistic review: verify the fixes are correct and complete, assess whether the "
        "tests provide adequate coverage, identify any remaining risks, and give a final "
        "verdict on whether the code is ready for production. Be critical and thorough. "
        "Conclude with a clear GO/NO-GO recommendation with reasoning."
    ),
    "SecurityAuditor": (
        "You are a security auditor specializing in application security. You will receive "
        "code changes and their descriptions. Your job is to check for security vulnerabilities "
        "including but not limited to: injection attacks (SQL, command, etc.), authentication "
        "and authorization flaws, data exposure risks, input validation issues, and insecure "
        "dependency usage. For each finding, classify the risk level (Critical/High/Medium/Low) "
        "using the CVSS framework. Do NOT fix the issues — only identify and describe them."
    ),
    "DocWriter": (
        "You are a technical documentation specialist. You will receive the final version of "
        "code changes, test results, and review findings. Your job is to produce clear, concise "
        "documentation covering: (1) what was changed and why, (2) how to use the modified code, "
        "(3) any caveats or known limitations, and (4) a changelog entry. Write for a technical "
        "audience but make it accessible. Use markdown formatting for clarity."
    ),
}

AGENT_ORDER = ["CodeAnalyst", "CodeFixer", "Tester", "Reviewer",
               "SecurityAuditor", "DocWriter"]

TASK_PROMPT = (
    "Here is a Python function that processes user orders in an e-commerce system. "
    "Please work on this code:\n\n"
    "```python\n"
    "def process_order(cart, user_id, promo_code=None):\n"
    "    total = 0\n"
    "    for item in cart:\n"
    "        price = item['price']\n"
    "        qty = item['quantity']\n"
    "        total += price * qty\n"
    "    if promo_code:\n"
    "        if promo_code == 'SAVE10':\n"
    "            total = total * 0.9\n"
    "        elif promo_code == 'NEWUSER':\n"
    "            total = total - 20\n"
    "    tax = total * 0.08\n"
    "    total = total + tax\n"
    "    user = get_user_from_db(user_id)\n"
    "    if user['credit'] < total:\n"
    "        return {'status': 'error', 'message': 'Insufficient credit'}\n"
    "    save_order_to_db(user_id, cart, total)\n"
    "    return {'status': 'success', 'total': total}\n"
    "```"
)


# ============================================================
# 合成前缀 Padding（模拟长 system prompt，增加 KV cache 压力）
# ============================================================

def make_prefix_pad(num_tokens: int) -> str:
    """生成确定性 padding，重复技术关键词序列以保证跨轮次一致（可被 cache）。"""
    if num_tokens <= 0:
        return ""
    pad_words = [
        "system", "configuration", "parameter", "optimization", "performance",
        "benchmark", "throughput", "latency", "memory", "allocation",
        "scheduling", "priority", "queue", "buffer", "pipeline",
        "protocol", "interface", "component", "module", "framework",
        "architecture", "distributed", "parallel", "sequential", "concurrent",
        "validation", "verification", "monitoring", "logging", "tracing",
    ]
    repeats = (num_tokens // len(pad_words)) + 1
    return " ".join(pad_words * repeats)


# ============================================================
# vLLM 服务管理
# ============================================================

def start_vllm_server(model: str, port: int, gpu: int,
                      gpu_mem_util: float, extra_args: List[str],
                      log_path: str) -> subprocess.Popen:
    """启动 vLLM 推理服务"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", "32768",
        *extra_args,
    ]

    flog = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=flog,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    proc._flog = flog  # type: ignore
    proc._cmd = cmd  # type: ignore
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """停止 vLLM 服务"""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
    if getattr(proc, "_flog", None):
        proc._flog.close()


def wait_for_server(base_url: str, timeout: int = 300) -> None:
    """等待服务就绪"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=10)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"Server not ready within {timeout}s at {base_url}")


# ============================================================
# Benchmark 核心
# ============================================================

def get_kv_cache_stats(base_url: str) -> Dict[str, float]:
    """从 vLLM Prometheus metrics 获取 KV cache 统计"""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=5)
        if r.status_code != 200:
            return {}
        text = r.text
        hits = 0.0
        total = 0.0
        for line in text.split("\n"):
            if line.startswith("vllm:prefix_cache_hits_total"):
                hits = float(line.split()[-1])
            elif line.startswith("vllm:prefix_cache_queries_total"):
                total = float(line.split()[-1])
        if total > 0:
            return {"hits": hits, "total": total, "hit_rate": hits / total}
    except Exception:
        pass
    return {}


def call_agent(base_url: str, model: str, agent_name: str,
               system_prompt: str, user_message: str,
               max_tokens: int = 256) -> Dict[str, Any]:
    """调用单个 Agent，返回耗时和响应"""
    tic = time.perf_counter()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=300,
    )
    elapsed = time.perf_counter() - tic
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return {
        "agent": agent_name,
        "latency_s": elapsed,
        "output_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "output": content,
    }


def run_workflow_round(base_url: str, model: str, round_idx: int,
                       prev_outputs: Dict[str, str],
                       prefix_pad: str = "") -> tuple[Dict[str, str], List[Dict]]:
    """执行一轮完整工作流，返回 (新输出, 每步耗时记录)"""
    outputs: Dict[str, str] = {}
    records: List[Dict] = []

    for i, agent_name in enumerate(AGENT_ORDER):
        system_prompt = prefix_pad + "\n" + AGENT_SYSTEM_PROMPTS[agent_name]

        # 构造 user message
        if i == 0:
            user_msg = TASK_PROMPT
        else:
            prev_agent = AGENT_ORDER[i - 1]
            prev_output = outputs.get(prev_agent, "")
            user_msg = (
                f"Task:\n{TASK_PROMPT}\n\n"
                f"Output from {prev_agent}:\n{prev_output}"
            )

        record = call_agent(base_url, model, agent_name, system_prompt, user_msg)
        record["round"] = round_idx
        record["step"] = i
        records.append(record)
        outputs[agent_name] = record["output"]

    return outputs, records


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Agent Workflow KV Cache Benchmark")
    parser.add_argument("--model", default="/data/models/Qwen3-1.7B")
    parser.add_argument("--port", type=int, default=18002)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--tight-vram", action="store_true",
                        help="Force small KV cache pool to trigger evictions")
    parser.add_argument("--prefix-pad-tokens", type=int, default=0,
                        help="Pad each agent's system prompt with N tokens")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of workflow rounds to measure")
    parser.add_argument("--warmup-rounds", type=int, default=2,
                        help="Warm-up rounds (not measured)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--log-dir", default="/tmp/vllm_bench_logs")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    server_log = os.path.join(args.log_dir, "vllm_server.log")
    base_url = f"http://127.0.0.1:{args.port}"

    # ---- 启动服务 ----
    mem_util = args.gpu_memory_utilization
    if args.tight_vram:
        mem_util = 0.35  # 极度压缩 KV cache 池, 强制触发驱逐
    print(f"[1/4] Starting vLLM server (model={args.model}, GPU={args.gpu}, mem_util={mem_util})...")
    extra_args = []
    if args.tight_vram:
        extra_args.extend(["--max-num-seqs", "8"])  # 允许多请求排队, 增加压力
    proc = start_vllm_server(args.model, args.port, args.gpu,
                             mem_util, extra_args, server_log)
    try:
        wait_for_server(base_url)
        print("       Server ready.")

        # ---- 生成 prefix padding ----
        prefix_pad = make_prefix_pad(args.prefix_pad_tokens)
        if args.prefix_pad_tokens > 0:
            print(f"       Prefix pad: ~{args.prefix_pad_tokens} tokens per agent")

        # ---- Warm-up ----
        print(f"[2/4] Warming up ({args.warmup_rounds} rounds)...")
        prev = {}
        for r in range(args.warmup_rounds):
            prev, _ = run_workflow_round(base_url, args.model, r, prev, prefix_pad)
            print(f"       Warm-up round {r + 1}/{args.warmup_rounds} done")

        # 读取 warm-up 后的 KV cache 基准
        stats_before = get_kv_cache_stats(base_url)

        # ---- 测量 ----
        print(f"[3/4] Measuring ({args.rounds} rounds)...")
        all_records: List[Dict] = []
        round_latencies: List[float] = []
        prev = {}
        for r in range(args.rounds):
            tic = time.perf_counter()
            prev, records = run_workflow_round(base_url, args.model, r, prev, prefix_pad)
            round_time = time.perf_counter() - tic
            round_latencies.append(round_time)
            all_records.extend(records)
            print(f"       Round {r + 1}/{args.rounds}: {round_time:.2f}s")

        stats_after = get_kv_cache_stats(base_url)

        # ---- 汇总 ----
        print("[4/4] Results:")
        avg_round = sum(round_latencies) / len(round_latencies)
        avg_per_agent = {
            name: sum(r["latency_s"] for r in all_records if r["agent"] == name)
            / sum(1 for r in all_records if r["agent"] == name)
            for name in AGENT_ORDER
        }

        result = {
            "model": args.model,
            "rounds": args.rounds,
            "avg_round_latency_s": round(avg_round, 2),
            "total_latency_s": round(sum(round_latencies), 2),
            "per_round_latency_s": [round(x, 2) for x in round_latencies],
            "avg_per_agent_latency_s": {k: round(v, 2) for k, v in avg_per_agent.items()},
            "kv_cache_before": stats_before,
            "kv_cache_after": stats_after,
        }

        print(json.dumps(result, indent=2, ensure_ascii=False))

        out_path = os.path.join(args.log_dir, "benchmark_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")

    finally:
        stop_server(proc)
        print("Server stopped.")


if __name__ == "__main__":
    main()
