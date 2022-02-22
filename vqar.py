# Toshiaki Koike-Akino, 2022
# QHack 2022, variational quantum autregressive (VQAR)
# train/test vqc.py given 2D-data
import pennylane as qml
import pennylane.numpy as np
import argparse
import sklearn
from sklearn import model_selection
from tqdm import tqdm
import vqc

# args
def get_args():
    parser = argparse.ArgumentParser('vqar')
    # general args
    parser.add_argument('--verb', action='store_true', help='verbose')

    # vqar args
    add_args(parser)

    return parser.parse_args()

# add args 
def add_args(parser):
    # simulator args
    sim_args(parser)

    # optimizer args
    opt_args(parser)

    # QML args
    vqc.vqc_args(parser)
    parser.set_defaults(skip=True)    

# sim args
def sim_args(parser):
    parser = parser.add_argument_group('sim')
    parser.add_argument('--split', default=0.9, type=float, help='train/test split ratio')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--epoch', default=100, type=int, help='training epoch')

# opt args
def opt_args(parser):
    parser = parser.add_argument_group('opt')
    parser.add_argument('--opt', default='AdamOptimizer', 
                        choices=['AdamOptimizer', 'AdagradOptimizer', 'GradientDescentOptimizer', 
                                 'MomentumOptimizer', 'NesterovMomentumOptimizer', 'RMSPropOptimizer', 'QNGOptimizer', 
                                 #'ShotAdaptiveOptimizer', 'LieAlgebraOptimizer', 'RotosolveOptimizer', 'RotoselectOptimizer',
                                 ],
                        help='optimizer. (some are not tested)')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (stepsize)')
    

# random seed
def seeding(args, verb=False):
    if args.seed > 0: # reseed if seed > 0
        if verb: print('# seeding', args.seed)
        np.random.seed(args.seed)
    else: # random
        args.seed = None 

# delay line unfolding: x[n], x[n-1], x[n-2], x[n-M], ...
def delay_line(data, memory=1):
    X = list()
    for k in range(memory + 1):
        X.append(np.roll(data, k, axis=0))
    X = np.hstack(X) # data:[N, D] -> X:[N, (M+1) D]
    X = X[memory:] # truncate rolled-up head
    #print(data)
    #print(X)
    return X

# optimizer selction
def get_opt(args):
    opt = getattr(qml, args.opt)(stepsize=args.lr)
    return opt

# model
def get_model(args, verb=False):
    if args.model == None:
        model = vqc.VQC(dev=args.dev, ansatz=args.ansatz, obs=args.obs,
                        dim=args.dim, layer=args.layer, memory=args.memory,
                        skip=args.skip, reup=args.reup, 
                        prescale=args.prescale, postscale=args.postscale, scalebias=args.scalebias,
                        verb=args.verb)
    else:
        model = vqc.load_model(fname=args.model, verb=verb)
    if verb: print('# model:', model)

    return model

# test
def testing(model, xtest, ytest):
    loss_ave = 0
    for x, y in tqdm(zip(xtest, ytest), leave=True, desc='test'):
        # predict
        yhat = model(x)

        # MSE loss
        loss = np.mean((y - yhat)**2)

        # loss average
        loss_ave += loss.item() / len(ytest)
    return loss_ave


# main process
def main(args, data):
    if args.verb: print('#', args)

    # seed
    seeding(args, verb=args.verb)

    # data shape [N, D]
    args.sample, args.dim = data.shape # orignal data shape
    
    # unfold
    data = delay_line(data, args.memory)
    if args.verb: print('# unfolded', data.shape)

    # train/test split
    train, test = sklearn.model_selection.train_test_split(data, train_size=args.split, shuffle=False)
    if args.verb: print('# train/test split', args.split, train.shape, test.shape)

    # model
    model = get_model(args, verb=args.verb)
    #model.draw()

    # optimizer
    opt = get_opt(args)
    if args.verb: print('# optimizer', opt)

    # cost
    def cost(weights, scales, x, y):
        model.weights = weights
        model.scales = scales
        yhat = model(x)
        #yhat = x * np.sum(weights)
        loss = np.mean((y - yhat)**2)
        #print('x, y, yhat, loss:', x._value, y._value, yhat._value, loss._value)
        #print(weights._value)
        return loss

    # initial weights
    weights = model.weights
    if args.verb: print('#weights:', weights.shape, weights)

    scales = model.scales
    if args.verb: print('#scales:', scales.shape, scales)

    # train loop
    for epoch in tqdm(range(args.epoch), leave=True, desc='epoch'):
        # permutation index of train data
        perm = np.random.permutation(len(train))
        #if args.verb: print('# perm', perm)

        # target: y, input: x
        ytrain, xtrain = train[perm, :args.dim], train[perm, args.dim:]
        ytest, xtest = test[:, :args.dim], test[:, args.dim:]
        #if args.verb: print('# yx-train yx-test', ytrain.shape, xtrain.shape, ytest.shape, xtest.shape)

        # batch loop
        loss_ave = 0
        for x, y in tqdm(zip(xtrain, ytrain), leave=True, desc='train'):
            #print('pre-weights', model.weights.shape)
            params, loss = opt.step_and_cost(cost, weights, scales, x, y)
            #if args.verb: print('step:', params[0].shape, params[1].shape, params[2].shape, params[3].shape, loss)

            # update
            weights = params[0]
            scales = params[1]
            model.weights = weights
            model.scales = scales

            # loss average
            loss_ave += loss.item() / len(perm)
            #print(model.weights[0])

        # test
        loss_test = testing(model, xtest, ytest)

        # results
        results = (loss_ave, loss_test, # MSE
                   loss_ave**0.5, loss_test**0.5, # RMSE
                   10 * np.log10(loss_ave), 10 * np.log10(loss_test)) # MSE in dB

        print(epoch, *results)

    # save models
    vqc.save_model(model, fname=f'model{args.memory}.p', verb=args.verb)
    model.draw()

    # optimized model
    return model, results

# example use
if __name__ == '__main__':
    # args
    args = get_args()
    
    # toy data: Wiener process
    data = np.random.randn(100, 2) # 100-sample of 2-dim data
    data = np.cumsum(data, axis=0) 
    
    # main train/test
    model, results = main(args, data)

    print(*results)
