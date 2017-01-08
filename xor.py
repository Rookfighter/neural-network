'''
Created on 08 Jan 2017

@author: Fabian Meyer
'''

import sys
import neuralnetwork as nn

HIDDEN_LEN = 5

def build_network():

    network = nn.neural_network(3)

    # input layer
    in1 = network.create_neuron(0, [1, 0], 0.5, 'threshold')
    in2 = network.create_neuron(0, [0, 1], 0.5, 'threshold')

    # hidden layer
    huid = []
    for _ in range(HIDDEN_LEN):
        h = network.create_neuron(1, nn.rand_weights(2), 1, 'log')
        network.create_connection(in1, h)
        network.create_connection(in2, h)
        huid.append(h)

    # output layer
    out1 = network.create_neuron(2, nn.rand_weights(HIDDEN_LEN), 1, 'log')
    for uid in huid:
        network.create_connection(uid, out1)

    return network

if __name__ == '__main__':

    network = build_network()

    # train the network
    print('Training network ...')
    data = nn.load_bin_data('data/xor.txt')
    nn.back_propagation(network, data)
    print('Training finished!')

    # wait for input to test trained network
    try:
        print('Give me a number: ')
        line = sys.stdin.readline().strip()
        while line:
            invals = nn.str_to_vals(line)

            if len(invals) != 2:
                print('Error: 2 digits only!')
            else:
                outvals = network.update(invals)
                outvals = [round(v) for v in outvals]

                print('{} => {}'.format(
                    nn.vals_to_str(invals),
                    nn.vals_to_str(outvals)))

            print('Give me a number: ')
            line = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        print('')
