'''
Created on 08 Jan 2017

@author: Fabian Meyer
'''

import sys
import neuralnetwork as nn

def build_network():

    network = nn.neural_network(3)

    # input layer
    in1 = network.create_neuron(0, [1, 0], 0.5, 'threshold')
    in2 = network.create_neuron(0, [0, 1], 0.5, 'threshold')

    # hidden layer
    h1 = network.create_neuron(1, nn.rand_weights(2), 0.5, 'threshold')
    network.create_connection(in1, h1)
    network.create_connection(in2, h1)

    h2 = network.create_neuron(1, nn.rand_weights(2), 0.5, 'threshold')
    network.create_connection(in1, h2)
    network.create_connection(in2, h2)

    # output layer
    out1 = network.create_neuron(2, nn.rand_weights(2), 0.5, 'threshold')
    network.create_connection(h1, out1)
    network.create_connection(h2, out1)

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

                print('{} => {}'.format(
                    nn.vals_to_str(invals),
                    nn.vals_to_str(outvals)))

            print('Give me a number: ')
            line = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        print('')
