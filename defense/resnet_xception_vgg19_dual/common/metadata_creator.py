import json
import os
import shutil

from argparse import ArgumentParser


def run():
    parser = ArgumentParser()
    parser.add_argument('--task', required=True, type=str)
    args = parser.parse_args()
    json.dump({
      'type': args.task,
      'container': 'gcr.io/tensorflow/tensorflow:1.1.0',
      'container_gpu': 'gcr.io/tensorflow/tensorflow:1.1.0-gpu',
      'entry_point': 'run.sh',
    }, open('../metadata.json', 'w'))
    shutil.copy('run_%s.sh' % args.task, '../run.sh')


if __name__ == '__main__':
    run()
