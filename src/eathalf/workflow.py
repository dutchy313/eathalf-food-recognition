"""Train on train, select on validation, and evaluate the selected model on test."""
import json
from pathlib import Path

from .data import CLASSES, read_manifest

MODEL_NAMES = ('cnn', 'cnn_aug', 'mobilenet128', 'mobilenet224', 'efficientnet', 'resnet')


def make_dataset(data_dir, rows, size, batch_size=32, training=False, seed=42):
    import tensorflow as tf
    paths = [str(Path(data_dir) / r['path']) for r in rows]
    labels = [CLASSES.index(r['label']) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(tf.cast(image, tf.float32), (size, size))
        return image, label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(name):
    """Preprocessing is part of the saved model; input is RGB float32 [0,255]."""
    import tensorflow as tf
    from tensorflow.keras import layers, applications
    size = 128 if name in ('cnn', 'cnn_aug', 'mobilenet128') else 224
    inputs = layers.Input(shape=(size, size, 3), name='rgb_0_255')
    x = inputs
    if name == 'cnn_aug':
        x = layers.RandomFlip('horizontal')(x)
        x = layers.RandomRotation(.08)(x)
        x = layers.RandomZoom(.10)(x)
    if name.startswith('cnn'):
        x = layers.Rescaling(1 / 255.)(x)
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
    else:
        if name.startswith('mobilenet'):
            x = layers.Rescaling(1 / 127.5, offset=-1)(x)
            base = applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(size, size, 3))
        elif name == 'efficientnet':
            # Keras EfficientNetB0 includes its own rescaling; keep [0,255] here.
            base = applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(size, size, 3))
        elif name == 'resnet':
            # RGB -> BGR, then subtract ImageNet BGR means. Serializable built-in layers.
            x = layers.Concatenate(axis=-1)([x[..., 2:3], x[..., 1:2], x[..., 0:1]])
            x = layers.Rescaling(scale=1., offset=[-103.939, -116.779, -123.68])(x)
            base = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(size, size, 3))
        else:
            raise ValueError(f'Unknown model: {name}')
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(.4)(x)
    outputs = layers.Dense(len(CLASSES), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, size


def metrics(model, dataset):
    """Labels and predictions are collected from the same batch, preventing misalignment."""
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
    truth, predictions = [], []
    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        truth.extend(labels.numpy().tolist())
        predictions.extend(np.argmax(probabilities, axis=1).tolist())
    labels = list(range(len(CLASSES)))
    return {
        'n': len(truth), 'accuracy': float(accuracy_score(truth, predictions)),
        'weighted_precision': float(precision_score(truth, predictions, labels=labels, average='weighted', zero_division=0)),
        'weighted_recall': float(recall_score(truth, predictions, labels=labels, average='weighted', zero_division=0)),
        'weighted_f1': float(f1_score(truth, predictions, labels=labels, average='weighted', zero_division=0)),
        'macro_f1': float(f1_score(truth, predictions, labels=labels, average='macro', zero_division=0)),
        'per_class': classification_report(truth, predictions, labels=labels, target_names=list(CLASSES), output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(truth, predictions, labels=labels).tolist(),
    }


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def train(data_dir, run_dir, names=MODEL_NAMES, epochs=10, batch_size=32, seed=42):
    import platform
    import subprocess
    import sys
    import tensorflow as tf
    run_dir = Path(run_dir)
    if not names or len(set(names)) != len(names) or not set(names) <= set(MODEL_NAMES):
        raise ValueError('Choose unique supported model names.')
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch-size must be positive.')
    if (run_dir / 'training_config.json').exists():
        raise ValueError('Training already started here. Prepare a new run directory.')
    train_rows, val_rows = [read_manifest(run_dir / f'{part}.csv') for part in ('train', 'val')]
    # Test manifest is deliberately not read by model selection.
    save_json(run_dir / 'training_config.json', {
        'models': list(names), 'epochs': epochs, 'batch_size': batch_size, 'seed': seed,
        'selection_metric': 'validation_macro_f1', 'tensorflow': tf.__version__, 'python': platform.python_version(),
        'input': 'RGB float32 [0,255]; resize bilinear; preprocessing inside model',
    })
    (run_dir / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    scores = []
    for name in names:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        model, size = build_model(name)
        train_ds = make_dataset(data_dir, train_rows, size, batch_size, training=True, seed=seed)
        val_ds = make_dataset(data_dir, val_rows, size, batch_size)
        model_dir = run_dir / name
        model_dir.mkdir()
        checkpoint = model_dir / 'best.keras'
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ])
        # Explicitly load the saved checkpoint so the evaluated model is the exported model.
        model = tf.keras.models.load_model(checkpoint)
        result = metrics(model, val_ds)
        save_json(model_dir / 'history.json', {k: [float(v) for v in values] for k, values in history.history.items()})
        save_json(model_dir / 'validation_metrics.json', result)
        scores.append({'model': name, 'input_size': size, **{k: result[k] for k in ('accuracy', 'macro_f1', 'weighted_f1')}})
    champion = max(scores, key=lambda item: item['macro_f1'])
    save_json(run_dir / 'validation_comparison.json', scores)
    save_json(run_dir / 'selection.json', {
        'model': champion['model'], 'input_size': champion['input_size'],
        'selection_basis': 'highest validation macro F1; ties follow requested model order',
        'checkpoint': f'{champion["model"]}/best.keras', 'classes': list(CLASSES),
        'preprocessing': 'embedded; RGB float32 [0,255]',
    })
    print('Selected using validation only:', champion)


def evaluate_test(data_dir, run_dir):
    import tensorflow as tf
    run_dir = Path(run_dir)
    destination = run_dir / 'test_metrics.json'
    if destination.exists():
        raise ValueError('Test metrics already exist. Preserve them; do not repeatedly tune against test.')
    selection = json.loads((run_dir / 'selection.json').read_text())
    model = tf.keras.models.load_model(run_dir / selection['checkpoint'])
    dataset = make_dataset(data_dir, read_manifest(run_dir / 'test.csv'), selection['input_size'])
    result = metrics(model, dataset)
    result['model'] = selection['model']
    result['scope'] = 'whole-image classification on custom test split; not detection or nutrition estimation'
    save_json(destination, result)
    print(json.dumps({k: result[k] for k in ('model', 'n', 'accuracy', 'macro_f1', 'weighted_f1')}, indent=2))
