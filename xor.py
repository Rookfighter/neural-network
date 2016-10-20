'''
Created on 20 Oct 2016

@author: Fabian Meyer
'''

import neunet.neuron as neuron
import neunet.training as training
import sys

if __name__ == '__main__':
    network = neuron.NeuralNetwork(3)

    network.addNeuron(0, 'linear', [1], 0.5)
    network.addNeuron(0, 'linear', [1], 0.5)
    network.addNeuron(1, 'log', [1, 1], 1)
    network.addNeuron(2, 'log', [1, 1, 1], 1)

    network.addEdge(0, 2)
    network.addEdge(1, 2)
    network.addEdge(0, 3)
    network.addEdge(1, 3)
    network.addEdge(2, 3)

    trainer = training.XorTrainer(network, 0.1)
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
            print('-- {0}'.format(outVals))
    except KeyboardInterrupt:
        print('')
