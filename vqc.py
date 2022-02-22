# Toshiaki Koike-Akino, 2022
# VQAR: Variational Quantum Circuit (VQC) for Vector Auto-Regressive (VAR) model
import pennylane as qml
import pennylane.numpy as np
import argparse
#from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

# vqc args
def vqc_args(parser):
    parser = parser.add_argument_group('vqc')
    parser.add_argument('--dev', default='default.qubit', help='quantum device')
    parser.add_argument('--layer', default=3, type=int, help='number of quantum layers')
    parser.add_argument('--ansatz', default='SimplifiedTwoDesign', help='VQC ansatz')
    parser.add_argument('--obs', default='expval', choices=['expval', 'var'], help='measurement')
    parser.add_argument('--model', default=None, help='model file to load. None: no-loading')
    parser.add_argument('--skip', action='store_true', help='skip connect')
    parser.add_argument('--reup', default=1, type=int, help='reuploading stages')
    parser.add_argument('--memory', default=1, type=int, help='VQAR memory')
    
    parser.add_argument('--prescale', action='store_true', help='pre-scaling inputs of VQC')
    parser.add_argument('--postscale', action='store_true', help='post-scaling outputs of VQC')
    parser.add_argument('--scalebias', action='store_true', help='using bias for scaling')

    # defaults
    parser.set_defaults(skip=True)
    parser.set_defaults(prescale=True)
    parser.set_defaults(postscale=True)
    parser.set_defaults(scalebias=True)

# args example
def get_args():
    parser = argparse.ArgumentParser('vqc')
    # general args
    parser.add_argument('--verb', action='store_true', help='verbose')

    # QML args
    vqc_args(parser)

    return parser.parse_args()

# VQC model
class VQC():
    def __init__(self, dev='default.qubit', ansatz='SimplifiedTwoDesign', obs='expval',
                dim=1, layer=1, memory=1, skip=False, reup=1, 
                prescale=True, postscale=True, scalebias=True,
                verb=False):
        self.verb = verb # verbose
        self.dim = dim # input/output dim
        self.layer = layer # quanum layer per memory
        self.memory = memory # NARX memory
        self.skip = skip # skip connect
        self.reup = reup # reuploading stage
        
        self.prescale = prescale
        self.postscale = postscale
        self.scalebias = scalebias
        
        self.qubit = self.dim * self.memory # number of qubits
        self.wires = range(self.qubit) 

        # QPU device
        self.dev = qml.device(dev, wires=self.qubit)
        if self.verb: print('# dev:', self.dev)

        # QML ansatz
        self.ansatz = getattr(qml, ansatz)
        if self.verb: print('# ansatz:', self.ansatz)

        # observable 
        self.obs = getattr(qml, obs)
        if self.verb: print('# obs:', self.obs)

        # qnode
        self.qnode = qml.QNode(self.circuit, device=self.dev)

        # weights shape
        self.shape = list(self.ansatz.shape(self.layer, self.qubit)[1]) # weights
        self.shape.insert(0, self.reup) # [R, weights]
        if self.verb: print('# shape:', self.shape)

        self.weights = self.init_weights()
        if self.verb: print('# weights:', self.weights.shape)

        self.scales = self.init_scales()
        if self.verb: print('# scalers:', self.scales.shape)

    # init weights
    def init_weights(self):
        return np.random.randn(*self.shape, requires_grad=True)

    # init scales
    def init_scales(self):
        # inputs and outputs scaler; [In/Out, Scale/Bias, Q]
        return np.random.randn(2, 2, self.qubit, requires_grad=True)

    # quantum circuit; inputs: [M, Q], weights: list of weights
    def circuit(self, inputs):
        # memory-times inputs
        #for k in range(self.memory):
        #if self.verb: print(k, inputs[k].shape, self.weights[k].shape)
        for k in range(self.reup):
            self.ansatz(inputs, self.weights[k], wires=self.wires)
        return tuple([self.obs(qml.PauliZ(i)) for i in self.wires])

    # scaling inputs/outpus of VQC
    def scaling(self, x, scale, bias):
        if self.scalebias: # scaling and bias
            return x * scale + bias
        else: # scaling only
            return x * scale        

    # run
    def __call__(self, inputs):
        # inputs scaling and bias
        #print('inputs:', inputs.shape)
        x = self.scaling(inputs, self.scales[0, 0], self.scales[0, 1]) if self.prescale else inputs
        
        # outputs
        x = self.qnode(x)
        #print('outputs:', outputs.shape)

        # output scaling and bias
        x = self.scaling(inputs, self.scales[1, 0], self.scales[1, 1]) if self.postscale else x
        #print('outputs:', outputs.shape)

        # skip
        x = x + inputs if self.skip else x

        return x[:self.dim] # discard outputs (could fold)

    # draw
    def draw(self, inputs=None):
        drawer = qml.draw(self.qnode, expansion_strategy='device')
        if inputs == None: # random inputs
            inputs = np.random.randn(self.qubit)
        if self.verb: 
            print('# inputs:', inputs.shape, inputs)
            print('# weights:', self.weights.shape, self.weights)
            print('# scales:', self.scales.shape, self.scales)
        print(drawer(inputs))

        drawer = qml.draw_mpl(self.qnode)
        fig, _ = drawer(inputs)
        fig.savefig(f'model.pdf')
        if self.verb: print('# saving model.pdf')
        plt.close(fig)
        
        spec = qml.specs(self.qnode)(inputs)
        print('# spec', spec)

# save model
def save_model(model, fname='model.p', path='models', verb=False):
    # path
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, fname)
    if verb: print('# saving', fname)

    with open(fname, 'wb') as file:
        pickle.dump(model, file)

# load model
def load_model(fname='models/model.p', verb=False):
    # path
    if verb: print('# loading', fname)

    with open(fname, 'rb') as file:
        model = pickle.load(file)
    return model