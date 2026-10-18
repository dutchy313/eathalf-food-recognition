# EatHalf model card

**Status:** Research prototype; no trained classifier weights distributed.

**Developer:** Oludotun Babayemi. **Framework:** TensorFlow/Keras. **Candidate models:** shallow and augmented CNNs; frozen ImageNet MobileNetV2, EfficientNetB0, and ResNet50 backbones with classification heads.

**Intended use:** Food-image classification experiments and evaluation of candidate methods for assisted meal logging. Users should confirm predictions in any later logging interface.

**Input/output:** RGB image resized to 128 or 224 pixels according to model; probability distribution over 17 alphabetically ordered food classes. Preprocessing is embedded in each corrected saved model. A single supported food category is assumed. Prediction forces a category even for non-food or unsupported images.

**Data:** Kaggle-sourced 17-class subset of Food-101; exact Kaggle listing unrecorded. Historical counts: 16,256 images; apple pie underrepresented. New runs report actual valid unique counts after quality checks. Custom stratified 70/15/15 split, not the official Food-101 split.

**Recorded historical result:** EfficientNetB0 accuracy 83.97%, weighted F1 0.8384 on 2,439 test images. Corrected experiments have not been trained or independently reproduced. Historical comparisons contain evaluation defects described in evaluation_notes.md.

**Corrected selection/evaluation:** Best validation-loss checkpoint per candidate; choose highest validation macro F1; evaluate selected model on test with accuracy, macro/weighted F1, weighted precision/recall, per-class metrics, and confusion matrix. Raw probabilities remain uncalibrated.

**Limits:** Restricted food/cuisine coverage, mixed dishes, occlusion, lighting changes, unsupported foods, class imbalance, duplicate leakage beyond exact pixel matches, and previously inspected test data. Claims of mobile readiness, retention improvement, portion estimation, calorie accuracy, ingredient identification, and health outcomes have not been validated.

**Product boundary:** Does not estimate food mass, nutrients, allergens, or medical suitability. A food category does not reveal recipes or hidden ingredients. Crop area is not a calibrated physical portion measure. Nutrition features require serving quantities, an appropriate food-composition source, user confirmation, and separate validation.

**Privacy:** No image upload service is implemented. Local prediction reads a user-selected file. If a hosted product is later introduced, define consent, retention, deletion, access controls, and treatment of photo metadata before collecting meals or health profiles.
