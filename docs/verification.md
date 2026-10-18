# Package verification

Prepared 18 October 2026.

## Completed checks

- Python source compilation and CLI help invocation succeeded.
- Three automated tests passed: deterministic/disjoint split generation with duplicate and unreadable-image exclusions; conflicting duplicate-label rejection; evaluation label/prediction alignment across deliberately reordered batches.
- Both notebooks passed nbformat schema validation; corrected notebook cells passed Python syntax parsing.
- All notebook outputs and execution counts are cleared; inline base64 images are removed.
- A targeted credential-pattern scan found no matching common API-token patterns. This is not an exhaustive secret audit.
- Historical EfficientNetB0 metrics were transcribed from the supplied notebook outputs; 2,048 correct plus 391 incorrect equals 2,439 test images.

## Not completed here

TensorFlow is not installed in the preparation environment. The optional TensorFlow preprocessing/save-load test was skipped. Full environment installation, ImageNet backbone downloads, model construction/training, GPU execution, inference from a trained classifier, YOLO downloads, and corrected numerical results have not been verified here. No dataset images or trained EatHalf weights were available for an end-to-end run.

After setup, enable the optional backbone test using `EATHALF_TEST_BACKBONES=1` and run `python -m pytest -q tests/test_preprocessing.py`. This downloads pretrained backbone weights. Then run the corrected workflow with your local data. Keep historical and new results separate.

The GitHub Actions job intentionally runs lightweight data/metric tests without TensorFlow. A green job does not demonstrate model-training or deployment readiness.
