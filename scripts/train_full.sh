#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
bash scripts/check_dataset_size.sh ../../datasets/celeba || {
  code=$?; [[ $code -eq 3 ]] && echo "Size gate skip" && exit 3; exit "$code"
}
bash scripts/train.sh configs/default.yaml
bash scripts/eval.sh configs/default.yaml
bash scripts/report.sh configs/default.yaml
echo "Full train + eval + report done"
