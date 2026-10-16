#!/usr/bin/env bash
set -euo pipefail
python -m src.train --epochs "${EPOCHS:-5}" --batch-size "${BATCH_SIZE:-64}"
python -m src.evaluate --checkpoint artifacts/best_model.pt --ood-datasets svhn cifar100
