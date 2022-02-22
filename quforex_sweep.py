# Toshiaki Koike-Akino
# sweep params for quforex

import forex_loader as loader
import argparse
import sweep_vqar

# args to inherit
def add_args(parser):
    # inherit sweep args
    sweep_vqar.add_args(parser)

    # inherit loader args
    loader.add_args(parser)

def get_args():
    parser = argparse.ArgumentParser('sweep')
    # general args
    parser.add_argument('--verb', action='store_true', help='verbose')
    #parser.set_defaults(verb=True)
    
    # add args
    add_args(parser)
    
    # defaults
    parser.set_defaults(country=['Canada'])
    parser.set_defaults(data='annual')
    parser.set_defaults(epoch=10)
    parser.set_defaults(skip=True)

    return parser.parse_args()

# main
def main(args):
    # data
    data = loader.main(args, drop_date=True, verb=args.verb).to_numpy()
    
    # sweep
    results = sweep_vqar.main(args, data)
    
    return results

#
if __name__ == '__main__':
    args = get_args()
    if args.verb: print(args)
    
    results = main(args)
    
    
    