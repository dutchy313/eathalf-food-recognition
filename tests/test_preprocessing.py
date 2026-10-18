"""Optional real TensorFlow serialization check; requires backbone downloads."""
import os
import pytest


def test_preprocessing_and_save_load(tmp_path):
    tf = pytest.importorskip('tensorflow')
    if os.environ.get('EATHALF_TEST_BACKBONES') != '1':
        pytest.skip('Set EATHALF_TEST_BACKBONES=1 to allow ImageNet downloads and backbone tests.')
    import numpy as np
    from eathalf.workflow import build_model
    for name in ('mobilenet128', 'efficientnet', 'resnet'):
        model, size = build_model(name)
        sample = tf.ones((1, size, size, 3)) * 128
        expected = model(sample, training=False).numpy()
        path = tmp_path / f'{name}.keras'
        model.save(path)
        restored = tf.keras.models.load_model(path)
        np.testing.assert_allclose(expected, restored(sample, training=False).numpy(), atol=1e-6)
        if name == 'mobilenet128':
            preprocessing = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling))
            np.testing.assert_allclose(preprocessing(tf.constant([0., 255.])).numpy(), [-1., 1.], atol=1e-6)
        if name == 'resnet':
            preprocessing = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling))
            np.testing.assert_allclose(preprocessing(tf.constant([[[[30., 20., 10.]]]])).numpy(), [[[[30-103.939, 20-116.779, 10-123.68]]]], atol=1e-5)
        tf.keras.backend.clear_session()
