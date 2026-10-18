# Third-party notices

EatHalf project code and documentation are offered under GNU AGPL-3.0. Copyright (c) 2026 Oludotun Babayemi. Third-party packages, pretrained weights, and images remain governed by their respective terms. This notice does not relicense them.

- **Ultralytics YOLO:** optional detection code uses the `ultralytics` package and downloads pretrained `yolo11n.pt`. Ultralytics publishes AGPL-3.0 and Enterprise licensing routes. The repository adopts AGPL-3.0 for the combined published research package. Review the terms for the particular code and weights before proprietary integration: https://www.ultralytics.com/license and https://github.com/ultralytics/ultralytics .
- **TensorFlow/Keras:** used for training and inference. Keep upstream notices and review the licenses of downloaded ImageNet weights: https://github.com/tensorflow/tensorflow and https://keras.io/api/applications/ . No backbone weights are bundled.
- **Food-101:** Bossard, Guillaumin, and Van Gool (2014), ETH Zurich. Dataset homepage: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ . The author's copy was obtained through Kaggle, but the exact listing is unresolved. No dataset photographs are included in this package. Dataset/image rights are separate from the repository license.
- Other dependencies include NumPy, Pillow, scikit-learn, Matplotlib, Jupyter, and pytest. Their licenses remain applicable; dependency files do not replace their notices.

If you plan to keep a later EatHalf product proprietary, resolve the Ultralytics licensing choice before integrating its code/models. Repository licensing does not establish permission for commercial use of the dataset images.
