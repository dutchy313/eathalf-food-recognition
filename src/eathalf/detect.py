"""Optional YOLO exploration; detector scores are not measured detection accuracy."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--output-dir', default='artifacts/detection-demo')
    parser.add_argument('--confidence', type=float, default=.4)
    args = parser.parse_args()
    from ultralytics import YOLO
    from PIL import Image
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Output exists; choose a new directory.')
    model = YOLO('yolo11n.pt')
    result = model(args.image, conf=args.confidence, verbose=False)[0]
    output.mkdir(parents=True)
    result.save(filename=str(output / 'detections.jpg'))
    with Image.open(args.image) as source:
        source = source.convert('RGB')
        for i, box in enumerate(result.boxes):
            label = model.names[int(box.cls.item())]
            print(i, label, float(box.conf.item()))
            # Limit exported crops to COCO's explicit food categories, excluding dishes/utensils.
            if label in {'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'}:
                source.crop(tuple(box.xyxy[0].cpu().tolist())).save(output / f'food_crop_{i}.jpg')
    print('Pretrained COCO detector: partial food coverage. Crops require separate classifier evaluation.')


if __name__ == '__main__':
    main()
