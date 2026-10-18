# Evaluation corrections and interpretation

## Historical record

The cleaned original notebook is retained for transparency. Its executable cells have not been presented as a repaired workflow. Personal Drive paths are replaced with generic paths, and all execution outputs, execution counts, cell metadata, attachments, and inline base64 images are removed. Use the new workflow notebook and Python modules for corrected experiments.

The original saved EfficientNetB0 test outputs agree internally: 2,048 correct and 391 incorrect predictions among 2,439 images, accuracy 0.8397, weighted precision 0.8399, weighted recall 0.8397, weighted F1 0.8384. These remain historical values, not newly reproduced scores.

## Corrections implemented in the new workflow

| Original issue | New behavior |
| --- | --- |
| Shuffled train/validation predictions compared against fixed directory labels | Evaluation collects labels and predictions from each same batch; validation order is deterministic |
| MobileNetV2 input scaled to [0,1] | Embedded scaling maps raw RGB [0,255] to [-1,1] |
| ResNet50 tested with [0,1] despite different training preprocessing | Embedded RGB-to-BGR conversion and ImageNet mean subtraction apply in training and inference |
| EfficientNet risk of double normalization | Raw [0,255] inputs feed the Keras backbone's internal preprocessing |
| Model selection labelled as test-based champion selection | Checkpoint per candidate chosen by validation loss; candidate selected by validation macro F1 |
| Local directories could retain older split files | Manifests reference original images; preparation refuses an existing output directory |
| Unsorted file enumeration affected repeatability | Sorted paths, seeded splits/training, explicit class tuple, and recorded environment |
| Image integrity checks limited to sample crops | All supported source images are decoded/verified before split; exact pixel duplicates removed |
| Detection confidence described as detection performance | Detection demo is explicitly exploratory; no box-grounded accuracy is claimed |

No corrected numerical results are fabricated. Rerun all candidates to assess their comparative performance. This workflow is a refactored experiment, not an exact replay of every historical architecture and run setting.

## Remaining limits

A newly randomized split of already inspected data is not a new independent external evaluation. Reusing the original images does not remove previous exposure. Preserve historical disclosure and collect a truly untouched external set before making stronger generalization claims. Keep people/devices/near-duplicate groups together if future datasets contain correlated images; the current preparation only detects exact decoded RGB duplicates.

The optional YOLO stage and whole-image classifier are separate experiments. Food-101 class labels alone cannot establish detection box quality. Measure end-to-end performance with annotated detection/cropping data, including missed detections, unrelated boxes, multiple foods, unsupported classes, and changes in classifier accuracy after cropping.

Preprocessing, split integrity, and metric logic can be checked without a full GPU training run. End-to-end installation, training, numerical reproducibility, model export, calibration, and deployment remain to be validated in the chosen environment.
