#!/usr/bin/env bash
cd "$(dirname "$0")"
export LLM_API_KEY="${TOGETHER_API_KEY}"
echo "start $(date -u)" > gen_s1.log
python3 -u gen_paraphrases.py --templates probe_templates_s1.json --out probes_s1_raw.json \
  --base-url https://api.together.xyz/v1 --model meta-llama/Llama-3.3-70B-Instruct-Turbo --n 5 --seed 20260928 >> gen_s1.log 2>&1
echo "exit $? $(date -u)" >> gen_s1.log
