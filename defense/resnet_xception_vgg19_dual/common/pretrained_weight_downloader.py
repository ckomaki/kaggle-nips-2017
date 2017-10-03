from argparse import ArgumentParser
from pretrianed_model_definition import PretrainedModelDefinition


def run():
    parser = ArgumentParser()
    parser.add_argument('--home-dir', dest='home_dir', type=str)
    parser.add_argument('--current-dir', dest='current_dir', type=str)
    parser.add_argument('--model-names', dest='model_names', type=str, nargs='+')
    args = parser.parse_args()
    print("home dir: %s" % args.home_dir)
    print("current dir: %s" % args.current_dir)
    for model_name in args.model_names:
        if 'nope' in model_name:
            continue
        print("download: %s" % model_name)
        PretrainedModelDefinition(model_name).download_model(args.home_dir, args.current_dir)


if __name__ == '__main__':
    run()
