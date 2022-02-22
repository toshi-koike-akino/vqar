# Toshiaki Koike-Akino
# sweep params to run quforex.py
import vqar
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

# sweep args
def sweep_args(parser):
    parser = parser.add_argument_group('sweep')
    parser.add_argument('--sweep', default=[1, 2, 3, 4], nargs='+', type=int, help='range')
    parser.add_argument('--target', default='memory', choices=['memory', 'layer', 'reup'], help='sweep target')
    
# add args to inherit
def add_args(parser):
    # sweep args
    sweep_args(parser)
    
    # vqar args
    vqar.add_args(parser)
    
# example args
def get_args():
    parser = argparse.ArgumentParser('sweep')
    # general args
    parser.add_argument('--verb', action='store_true', help='verbose')
    #parser.set_defaults(verb=True)
    parser.add_argument('--size', default=[100, 2], type=int, nargs=2, help='toy data size')
    parser.add_argument('--data', default='toy', type=str, help='results name')
    
    # add args
    add_args(parser)
    
    # default args
    parser.set_defaults(skip=True)
    
    return parser.parse_args()

# sweeping 
def sweep(args, data):
    results_all = list()
    for s in args.sweep:
        # sweep target value
        if args.target == 'layer':
            args.layer = s
        elif args.target == 'memory':
            args.memory = s
        elif args.target == 'reup':
            args.reup = s
        else:
            raise ValueError(args.target, args.sweep)

        # run quforex
        _, results = vqar.main(args, data)
                
        # results
        results = list((s, *results))
        print(*results)
        
        results_all.append(results)
                
    return results_all
                
# plot results
def plot(results, xlabel=None, title=None, path='plots'):
    plt.figure()    
    plt.plot(results[:,0], results[:,5], label='train')
    plt.plot(results[:,0], results[:,6], label='test')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('MSE (dB)')
    plt.title(title)
    plt.grid()
    
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'{title}.png'))
    plt.show()
                
# save results
def save(results, fname='results.npy', path='results'):
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, fname)
    with open(fname, 'wb') as file:
        np.save(file, results)

# main
def main(args, data):
    # sweep
    results = np.stack(sweep(args, data))
    print(results)    
    print(results.shape)

    title = f'{args.data}_sweep-{args.target}_{args.memory}_{args.layer}_{args.reup}'
    # save results    
    save(results, fname=f'{title}.npy')

    # plot results
    plot(results, xlabel=args.target, title=title)
    
    return results


if __name__ == '__main__':
    args = get_args()
    if args.verb: print(args)
    
    # toy data: Wiener process
    data = np.random.randn(*args.size) # 100-sample of 2-dim data
    data = np.cumsum(data, axis=0) 
    
    # sweep
    results = main(args, data)
    
    print(results)
    
    