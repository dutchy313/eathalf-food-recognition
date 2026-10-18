# EatHalf AI
### Food Image Recognition for Dietary Tracking

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue)](LICENSE)
![Status](https://img.shields.io/badge/Status-Research%20prototype-orange)

EatHalf explores how computer vision can reduce the effort of identifying foods when logging meals. The project compares convolutional neural networks and transfer learning models on a **17-category subset of Food-101**, with a separate exploration of pretrained YOLO detection and cropping.

**Author:** [Oludotun Babayemi](https://github.com/dutchy313)

> **Current scope:** image classification research and experimental object detection. Portion size, calories, nutrients, personalized dietary recommendations, and a deployed meal tracker are future work. The corrected workflow has not yet been trained end to end; the results below come from the original notebook.

## What this repository contains

- Dataset inspection, class balance analysis, and qualitative error analysis in the historical notebook.
- Six configurable experiments: shallow CNN, augmented CNN, MobileNetV2 at 128 and 224 pixels, EfficientNetB0, and ResNet50.
- A corrected training workflow with embedded model-specific preprocessing, stable class ordering, and aligned evaluation labels.
- Custom stratified splits with image validation, exact RGB-pixel duplicate removal, and saved manifests.
- Selection by validation macro F1, followed by test evaluation of the selected model.
- Whole-image top-three prediction and an optional pretrained YOLO11 detection demo.

## Recorded results from the original notebook

| Measure | Saved result |
| --- | ---: |
| Dataset size | 16,256 images |
| Food categories | 17 |
| Custom test set | 2,439 images |
| EfficientNetB0 test accuracy | 83.97% |
| EfficientNetB0 weighted precision | 0.8399 |
| EfficientNetB0 weighted recall | 0.8397 |
| EfficientNetB0 weighted F1 | 0.8384 |
| Correct / incorrect predictions | 2,048 / 391 |

These are **historical, unreplicated whole-image classification results**. They do not measure detection accuracy, calorie estimation, or the combined detection-and-classification pipeline. The original validation comparison has prediction-order problems, and the original ResNet50 test comparison uses inconsistent preprocessing. Those comparisons should not guide model selection. See [evaluation notes](docs/evaluation_notes.md) and the [model card](docs/model_card.md).

New runs will produce new results. Deduplication, corrected preprocessing, and different software/hardware can change the sample counts and scores. No pretrained EatHalf classifier weights are included.

## Setup in VS Code

Recommended starting environment: **Python 3.11**, a virtual environment, and the VS Code Python and Jupyter extensions. The declared dependency ranges are a starting configuration, not a tested lockfile. Each training run records its installed packages in `environment.txt`.

Open this repository's root folder in VS Code. In a PowerShell terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell activation is blocked, use `.\.venv\Scripts\python.exe` in place of `python`; activation is optional. Select that environment using **Python: Select Interpreter** and select the same kernel for notebooks.

On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A GPU is recommended for comparing all six models. Windows CPU execution is possible but can be slow. For GPU training, use a compatible Linux/WSL2 TensorFlow environment or a GPU notebook runtime. This project does not assert mobile latency or offline deployment performance.

## Get the data

Download Food-101 independently from the Kaggle source used for your experiment, or use the [original ETH Zurich dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). Dataset images are not bundled or relicensed here. The exact Kaggle listing for the author's historical copy has not yet been recorded; see [data provenance](docs/data_sources.md).

Point `--data-dir` at a directory containing these 17 class folders. For the full original dataset, point it at `food-101/images`; only the listed classes are used.

```text
local-food-data/
  apple_pie/
  chocolate_cake/
  donuts/
  falafel/
  french_fries/
  hot_dog/
  ice_cream/
  nachos/
  onion_rings/
  pancakes/
  pizza/
  ravioli/
  samosa/
  spring_rolls/
  strawberry_shortcake/
  tacos/
  waffles/
```

Keep data outside Git, or under the ignored `data/` folder. The historical apple-pie folder had only 256 images; a fresh full Food-101 copy may have 1,000. Do not remove valid images to reproduce the old count.

## Run the corrected workflow

Replace the example path with your own. Run commands from the repository root.

```powershell
python -m eathalf.cli prepare --data-dir "C:/datasets/Food_101" --run-dir artifacts/run-001
python -m eathalf.cli train --data-dir "C:/datasets/Food_101" --run-dir artifacts/run-001 --models efficientnet --epochs 10
python -m eathalf.cli test --data-dir "C:/datasets/Food_101" --run-dir artifacts/run-001
python -m eathalf.predict --image "C:/photos/meal.jpg" --run-dir artifacts/run-001
```

To compare all six models, replace the training command with:

```powershell
python -m eathalf.cli train --data-dir "C:/datasets/Food_101" --run-dir artifacts/run-001 --models cnn cnn_aug mobilenet128 mobilenet224 efficientnet resnet --epochs 10
```

Choose the candidate models before examining test results. Training uses the best validation-loss checkpoint for each candidate, then selects the candidate with the highest validation macro F1. Test evaluation reads only the selected checkpoint. The command refuses to overwrite existing test results; this is a practical guard, not proof of an untouched test set. The original notebook's test set has already been examined. Use independently collected, untouched images for a stronger final assessment.

Run outputs include split manifests, exclusion logs, class counts, environment details, model checkpoints, training histories, validation metrics, selected-model metadata, and test metrics with per-class scores and a confusion matrix. Outputs stay under ignored `artifacts/`. Review and copy selected, non-sensitive metrics into `reports/` when you publish new results.

The guided version is [notebooks/eathalf_reproducible_workflow.ipynb](notebooks/eathalf_reproducible_workflow.ipynb). The [historical notebook](notebooks/eathalf_original_exploration.ipynb) preserves the author's exploration and narrative, with outputs and embedded images removed. It is reference material, not the corrected entry point.

## Optional YOLO exploration

```powershell
python -m pip install -r requirements-detection.txt
python -m eathalf.detect --image "C:/photos/meal.jpg"
```

This downloads `yolo11n.pt` on first use and uses pretrained COCO classes, which cover only some foods. The demo prints detected classes and exports crops only for explicit COCO food labels. It does not train a 17-class detector or automatically validate the crops. No-detection and non-food cases remain product-development requirements. Detector confidence is not detection accuracy; evaluating detection requires suitable box annotations and metrics.

## Limitations and roadmap

- The 17 classes provide limited cuisine coverage, including limited representation of Nigerian and other African meals.
- Whole-image classification assumes a supported dominant food category. Mixed meals and unfamiliar foods need additional handling.
- Softmax scores are uncalibrated and are not a reliable measure of certainty.
- RGB pixel hashing removes exact decoded duplicates; visually similar near-duplicates still require review.
- Meal volume, mass, ingredients, allergens, and nutrients cannot be established from the food label or crop area alone.
- Next steps: rerun corrected experiments, investigate apple-pie provenance, test external images, calibrate predictions, evaluate cropping end to end, then add nutrition lookup with user-confirmed serving quantities.
- Mobile export, device benchmarks, privacy-conscious meal logging, and a user interface are planned, not implemented.

## Verification

```powershell
python -m pytest -q
python -m compileall -q src
```

The lightweight tests check split integrity, duplicate handling, and metric alignment. An opt-in TensorFlow-dependent test checks preprocessing and save/load behavior when TensorFlow is installed and `EATHALF_TEST_BACKBONES=1` is set. See [verification details](docs/verification.md). Training and GPU performance require local verification.

## License and attribution

The repository is distributed under **GNU AGPL-3.0**; see [LICENSE](LICENSE). Copyright (c) 2026 Oludotun Babayemi. Third-party software, pretrained weights, and dataset images retain their own terms. The included optional Ultralytics workflow requires attention to Ultralytics' AGPL/Enterprise terms before proprietary integration. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Food-101: Bossard, L., Guillaumin, M., & Van Gool, L. (2014). *Food-101 – Mining Discriminative Components with Random Forests*. European Conference on Computer Vision.
