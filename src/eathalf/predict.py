"""Whole-image top-k prediction. Requires a locally trained run."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--run-dir', default='artifacts/run-001')
    args = parser.parse_args()
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    run = Path(args.run_dir)
    metadata = json.loads((run / 'selection.json').read_text())
    model = tf.keras.models.load_model(run / metadata['checkpoint'])
    with Image.open(args.image) as image:
        array = np.array(image.convert('RGB'), dtype='float32')
    tensor = tf.image.resize(array, (metadata['input_size'], metadata['input_size']), method='bilinear')
    probabilities = model(tensor[None, ...], training=False).numpy()[0]
    for index in np.argsort(probabilities)[::-1][:3]:
        print(f'{metadata["classes"][index]}: {probabilities[index]:.2%}')
    print('Scores are uncalibrated model probabilities; unfamiliar/non-food images can receive high scores.')


if __name__ == '__main__':
    main()
