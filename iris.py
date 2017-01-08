'''
Created on 8 Jan 2017

@author: fabian
'''

import sys
import neuralnetwork as nn

LAYER_LEN = 1
HIDDEN_LEN = 20

def build_network():

    network = nn.neural_network(LAYER_LEN + 2)

    # input layer
    network.create_neuron(0, [1, 0, 0, 0], 1, 'linear')
    network.create_neuron(0, [0, 1, 0, 0], 1, 'linear')
    network.create_neuron(0, [0, 0, 1, 0], 1, 'linear')
    network.create_neuron(0, [0, 0, 0, 1], 1, 'linear')

    # hidden layer
    for l in range(1, LAYER_LEN + 1, 1):
        lbc = len(network.layers[l - 1])
        for _ in range(HIDDEN_LEN):
            h = network.create_neuron(l, nn.rand_weights(lbc), 1, 'log')
            for uid in network.layers[l - 1]:
                network.create_connection(uid, h)

    # output layer
    lbc = len(network.layers[-2])
    out1 = network.create_neuron(LAYER_LEN + 1, nn.rand_weights(lbc), 1, 'log')
    out2 = network.create_neuron(LAYER_LEN + 1, nn.rand_weights(lbc), 1, 'log')
    out3 = network.create_neuron(LAYER_LEN + 1, nn.rand_weights(lbc), 1, 'log')
    for uid in network.layers[-2]:
        network.create_connection(uid, out1)
        network.create_connection(uid, out2)
        network.create_connection(uid, out3)

    return network

if __name__ == '__main__':

    network = build_network()

    # train the network
    print('Training network ...')
    data = nn.load_iris_data('data/iris.csv')
    nn.back_propagation(network, data, learnfac=0.5, min_err=1)
    print('Training finished!')

    # wait for input to test trained network
    try:
        data = nn.load_iris_data('data/iris2.csv')

        print('Give me a data number: ')
        line = sys.stdin.readline().strip()
        while line:
            ln = int(line)

            outvals = network.update(data[ln][0])

            print('{} = {}'.format(
                outvals,
                data[ln][1]))

            print('Give me a number: ')
            line = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        print('')
