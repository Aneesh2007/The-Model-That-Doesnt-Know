# The Model That Doesn't Know

An image classifier that predicts one of ten familiar classes **or abstains and returns `UNKNOWN`** when its uncertainty score crosses a threshold. The project deliberately evaluates both familiar and excluded image categories, rather than reporting classification accuracy alone.

## What is implemented

- **Known classes:** CIFAR-10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **OOD evaluation sets:** SVHN house-number digits (different visual domain) and CIFAR-100 test images (unseen fine-grained categories, some of which can be semantically or visually close to CIFAR-10).
- **Model:** ImageNet-pretrained ResNet-18 with a new 10-class head, fine-tuned on CIFAR-10.
- **Preprocessing:** RGB; resize to 128×128; ImageNet normalization. Training-only random crop, horizontal flip, and light color jitter. Validation and test transforms are deterministic.
- **Validation:** A seeded 80/10/10 split of the CIFAR-10 training set: train, model-selection validation, and separate calibration split. The test set is kept separate.
- **Uncertainty methods:** (1) maximum softmax probability (MSP), and (2) energy score, `E(x) = -T logsumexp(logits/T)`.
- **Calibration:** A scalar temperature is fit on the separate known-class calibration split. The confidence threshold is the 5th percentile of calibration-split confidence, and the energy threshold is the 95th percentile of calibration-split energy by default. Thus thresholds target roughly 5% known-class rejection on the calibration split; they do **not** use the OOD test sets.
- **Evaluation:** Top-1 accuracy, accuracy among accepted known images, known false-rejection rate, OOD rejection rate, OOD AUROC/AUPR, top-label ECE before/after temperature scaling, confusion matrix, per-sample decisions, and score-distribution plots.
- **Demo:** Optional Streamlit image upload UI.

## Setup

For the reproducible Windows setup, use Python 3.12.x (64-bit). Run commands from the project root. See [WINDOWS_SETUP_FIX.md](WINDOWS_SETUP_FIX.md) if you hit the NumPy build error on Python 3.14.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
# Windows CPU-only: install matching PyTorch wheels first
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The requirements are pinned to a known-compatible Python 3.12 setup. The commands above install the CPU-only PyTorch pair; choose the appropriate PyTorch index if you specifically need GPU support. The first run downloads the datasets and ImageNet weights, so an internet connection and several GB of free storage are recommended.

## Run the full pipeline

```bash
# 1. Train; downloads CIFAR-10 and pretrained ResNet-18 weights when needed.
python -m src.train --epochs 5 --batch-size 64

# 2. Evaluate on the untouched CIFAR-10 test set and two excluded datasets.
python -m src.evaluate --checkpoint artifacts/best_model.pt --ood-datasets svhn cifar100

# 3. Predict/reject a single image.
python -m src.predict --image path/to/image.jpg --method confidence
python -m src.predict --image path/to/image.jpg --method energy

# 4. Optional UI.
streamlit run app.py

# 5. Small logic tests.
python -m pytest -q
```

Useful flags:

```bash
python -m src.train --epochs 10 --batch-size 32 --image-size 128 --known-rejection 0.05
python -m src.evaluate --batch-size 64 --num-workers 0
```

On a CPU-only laptop, start with 3–5 epochs and batch size 16–32. `--no-pretrained` is available for experimentation, but random initialization needs substantially more training and is not the recommended baseline.

## How unknown detection works

The classifier always emits ten logits. By default, the raw prediction is the highest-scoring class. The rejector then applies a separate rule:

- **MSP:** compute the maximum temperature-scaled softmax probability. If it is below the confidence threshold, return `UNKNOWN`.
- **Energy:** compute `E(x) = -T logsumexp(logits/T)`. If energy is above the known-calibration threshold, return `UNKNOWN`.

Temperature scaling is learned from known calibration labels only. It preserves the argmax class but can change the probabilities. It improves probability calibration in some cases; it does not guarantee OOD detection. Both thresholds are fitted on known calibration data, making the target known-acceptance rate explicit. OOD evaluation therefore measures how well the learned scores generalize to excluded data, rather than tuning the threshold directly on test OOD examples.

## Outputs

Training outputs are saved under `artifacts/`:

- `best_model.pt`: model weights, class names, input size, temperature and thresholds.
- `training_history.json`: loss and accuracy per epoch.
- `thresholds.json`: calibrated score thresholds and temperature.

Evaluation outputs are saved under `artifacts/evaluation/`:

- `metrics.csv` and `report.json`: dataset and detection metrics.
- `known_confusion_matrix.csv`: CIFAR-10 test confusion matrix.
- `per_sample_predictions.csv`: raw prediction, confidence/energy, decisions, and known-set correctness for each sample.
- `confidence_distributions.png` and `energy_distributions.png`: known/OOD score overlap and threshold line.

**Do not copy imaginary scores into the submission.** Run the pipeline in your environment and report the values generated by your own experiment, plus the hardware, epoch count, and package versions. The project code is provided; its full training run has not been performed as part of generating these files.

## What to discuss in the report

1. How strong is normal CIFAR-10 top-1 accuracy before rejection?
2. At the chosen threshold, how many known images are rejected, and how many OOD images are rejected?
3. Does energy rank OOD images better than MSP? Compare AUROC and AUPR, not only one threshold.
4. Among accepted known images, does selective accuracy rise? What coverage was sacrificed?
5. Inspect `per_sample_predictions.csv`: examples of confident OOD false accepts and known-class false rejects.
6. Compare ECE before/after temperature scaling. Explain why better calibration and better OOD detection are related but not equivalent.
7. Compare SVHN with CIFAR-100. A model may detect digits fairly easily but confuse CIFAR-100 objects with semantically similar CIFAR-10 objects.

## Limitations and responsible interpretation

- OOD is relative to the training label space. These tests do not prove general open-world safety.
- Confidence and energy can both be high/low for the wrong reasons. Neural networks can be overconfident on unseen classes.
- The 5% target is approximate and only guaranteed on the calibration split up to sampling variation; test rejection rates can differ.
- CIFAR-10 and CIFAR-100 are small, low-resolution benchmark datasets. Real camera images can involve domain shift, blur, occlusion, lighting changes, and label ambiguity.
- This is a thresholded classifier, not a semantic novelty detector. An `UNKNOWN` result means “needs review,” not a definitive claim that the image belongs to no known class.
- Temperature scaling estimates calibration on the known calibration distribution and cannot repair all distribution shifts.
- Pretrained ImageNet weights may have been trained on related visual concepts; the final supervised head is still trained only on CIFAR-10 labels. Explain this choice honestly in the interview.

## Suggested Git commit sequence

1. `feat: add CIFAR data pipeline and ResNet18 baseline`
2. `feat: add MSP and energy-based rejectors`
3. `feat: add known-vs-OOD evaluation and plots`
4. `feat: add single-image and Streamlit demos`
5. `docs: document methodology, results, and failure cases`
