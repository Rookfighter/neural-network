'''
Created on 20 Oct 2016

@author: Fabian Meyer
'''

import neunet.neuron as neuron
import neunet.training as training
import sys
import random

def rndInWeights(c):
    return [ random.random() for _ in range(c)]

if __name__ == '__main__':
    network = neuron.NeuralNetwork(3)

    network.addNeuron(0, 'linear', [1], 0.5)
    network.addNeuron(0, 'linear', [1], 0.5)
    network.addNeuron(1, 'log', rndInWeights(2), 1)
    network.addNeuron(2, 'log', rndInWeights(3), 1)

    network.addEdge(0, 2)
    network.addEdge(1, 2)
    network.addEdge(0, 3)
    network.addEdge(1, 3)
    network.addEdge(2, 3)

    trainer = training.XorTrainer(network, 5, 0.3)
    trainer.load('training/xor.txt')
    trainer.train()

    try:
        while True:
            print('Give me a number: ')
            line = sys.stdin.readline().strip()
            if not line:
                break

            inVals = training.binStrToVals(line)
            inVals = [[val] for val in inVals]
            outVals = network.compute(inVals)
            print('-- {0}'.format(round(outVals[0])))
    except KeyboardInterrupt:
        print('')
