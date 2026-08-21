#!/bin/bash
set -e

echo "Downloading CrossMemBench..."
hf download devesht01/CrossMemBench \
    --repo-type dataset \
    --include "data/**" \
    --local-dir benchmarks/CrossMemBench

echo "Downloading PersonaMem..."
mkdir -p benchmarks/PersonaMem/data

hf download bowen-upenn/PersonaMem-v1 \
    questions_32k.csv \
    shared_contexts_32k.jsonl \
    --repo-type dataset \
    --local-dir benchmarks/PersonaMem/data

curl -L \
    https://raw.githubusercontent.com/bowen-upenn/PersonaMem/main/data/random_questions.txt \
    -o benchmarks/PersonaMem/data/random_questions.txt

curl -L \
    https://raw.githubusercontent.com/bowen-upenn/PersonaMem/main/data/random_code_questions.txt \
    -o benchmarks/PersonaMem/data/random_code_questions.txt

echo "Done."