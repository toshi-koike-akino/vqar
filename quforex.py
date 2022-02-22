# Toshiaki Koike-Akino, 2022
# QHack 2022, variational quantum autregressive (VQAR) for FX prediction
import argparse
import forex_loader as loader
import vqar

# args example
def get_args():
    parser = argparse.ArgumentParser('quforex')
    # general args
    parser.add_argument('--verb', action='store_true', help='verbose')

    # add args
    add_args(parser)
        
    return parser.parse_args()

# add args to inherit
def add_args(parser):
    # inherit vqar args
    vqar.add_args(parser)

    # inherit loader args
    loader.add_args(parser)
    
    # may change defaults
    parser.set_defaults(country=['Canada', 'Norway', 'Denmark', 'Sweden', 'Switzerland', 'New Zealand'])
    parser.set_defaults(data='annual')

# main process
def main(args):
    # data
    data = loader.main(args, drop_date=True, verb=args.verb).to_numpy()

    # train/test
    model, results = vqar.main(args, data)
    
    return model, results

# example use
if __name__ == '__main__':
    # args
    args = get_args()

    # main train/test
    model, results = main(args)

    print(*results)
