"""Run with python -m eathalf.cli --help."""
import argparse
from .data import prepare


def main():
    parser = argparse.ArgumentParser(description='EatHalf reproducible classification workflow')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'train', 'test'):
        task = sub.add_parser(command)
        task.add_argument('--data-dir', required=True, help='Folder containing the 17 food class folders')
        task.add_argument('--run-dir', default='artifacts/run-001')
        if command in ('prepare', 'train'):
            task.add_argument('--seed', type=int, default=42)
        if command == 'train':
            task.add_argument('--models', nargs='+', choices=['cnn', 'cnn_aug', 'mobilenet128', 'mobilenet224', 'efficientnet', 'resnet'], default=['efficientnet'])
            task.add_argument('--epochs', type=int, default=10)
            task.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.data_dir, args.run_dir, args.seed))
    else:
        from .workflow import train, evaluate_test
        if args.command == 'train':
            train(args.data_dir, args.run_dir, args.models, args.epochs, args.batch_size, args.seed)
        else:
            evaluate_test(args.data_dir, args.run_dir)


if __name__ == '__main__':
    main()
