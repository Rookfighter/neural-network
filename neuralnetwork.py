'''
Created on 25 Dec 2016

@author: Fabian Meyer
'''
import random
import math

def _threshold_func(x, threshold):
    if x >= threshold:
        return 1
    else:
        return 0

def _log_func(x, fac):
    return 1 / (1 + math.exp(-fac * x))

class neuron:

    def __init__(self, uid, weights, activity_param, activity_func):
        self.uid = uid
        self.weights = weights
        self.activity_param = activity_param
        self.outval = 0
        self.__activity_funcs = {
            'threshold' : _threshold_func,
            'log' : _log_func
        }
        self.__activity_func = self.__activity_funcs[activity_func]

    def __net_func(self, invals):
        return sum([w * v for w, v in zip(self.weights, invals)])

    def __output_func(self, actval):
        return actval

    def update(self, invals):
        '''
        Calculates the reaction of the neuron to the given input.

        @param invals: vector of input values
        '''

        assert(len(self.weights) == len (invals))

        netval = self.__net_func(invals)
        actval = self.__activity_func(netval, self.activity_param)
        self.outval = self.__output_func(actval)


class neural_network:

    def __init__(self, layer_count):
        '''
        Creates a new neural network with the given amount of layers.

        @param layer_count: amount of layers for the network
        '''

        self.neurons = []
        self.layers = [[] for _ in range(layer_count)]
        self.edges_ba = []
        self.edges_ab = []

    def create_neuron(self, layer, weights, activity_param=0.5, activity_func='threshold'):
        '''
        Creates a new neuron in the network in the given layer
        with the given weights and threshold.

        @param layer:      layer the neuron should occupy
        @param weights:    vector of weights for the neuron
        @param threshold:  activity threshold at which neuron should react

        @return: uid of the created neuron
        '''

        assert(layer >= 0 and layer < len(self.layers))

        uid = len(self.neurons)

        self.neurons.append(neuron(uid, weights, activity_param, activity_func))
        self.layers[layer].append(uid)
        self.edges_ab.append([])
        self.edges_ba.append([])

        return uid


    def create_connection(self, uid_a, uid_b):
        '''
        Creates a connection from neuron A (out) to neuron B (in).

        @param uid_a: uid of neuron A
        @param uid_b: uid of neuron B
        '''

        assert(uid_a >= 0 and uid_a < len(self.neurons))
        assert(uid_b >= 0 and uid_b < len(self.neurons))

        self.edges_ba[uid_b].append(uid_a)
        self.edges_ab[uid_a].append(uid_b)

    def __update_layer(self, layer):
        # go through all neurons in this layer
        for uid in layer:
            # create input vector
            invals = [self.neurons[i].outval for i in self.edges_ba[uid]]
            self.neurons[uid].update(invals)

    def update(self, invals):
        '''
        Update the neural network with the given invals and
        create output reaction of the network.

        @param invals: vector of input values

        @return: vector of output values
        '''

        assert(self.layers)
        assert(len(invals) == len(self.layers[0]))

        # first layer = input layer
        for uid in self.layers[0]:
            self.neurons[uid].update(invals)

        # go through remaining _layers
        for layer in self.layers[1:]:
            self.__update_layer(layer)

        # return outVals of last layer = output layer
        return [self.neurons[uid].outval for uid in self.layers[-1]]

def rand_weights(n):
    return [ random.random() for _ in range(n)]

def str_to_vals(s):
    return [int(c) for c in s]

def vals_to_str(v):
    return ''.join([str(i) for i in v])

def load_bin_data(filename):
    '''
    Loads training data from the given file.

    The result has the following format:

    [
        (invals1, outvals1),
        (invals2, outvals2),
        (invals3, outvals3),
        ...
    ]

    @param filename: path to the file with the training data

    @return: matrix of training data
    '''

    data = []
    with open(filename) as f:
        for l in f:
            invals, outvals = l.split()
            invals = str_to_vals(invals)
            outvals = str_to_vals(outvals)
            data.append((invals, outvals))

    return data

def delta_training(network, data, learnfac=0.05):
    '''
    Trains the given network with delta learning rule. The parameter
    data has to have the following format:

    [
        (invals1, outvals1),
        (invals2, outvals2),
        (invals3, outvals3),
        ...
    ]

    @param network: the network to be trained
    @param data: the data which is used to train the network
    '''

    def train_step():
        '''
        A single training step that goes through all data samples
        and applies the delta learning rule.
        '''
        for sample in data:
            outvals = network.update(sample[0])

            assert(len(sample[1]) == len(outvals))
            assert(len(network.layers[-1]) == len(outvals))
            # iterate through all output neurons and their results
            for exp, act, uid in zip(sample[1], outvals, network.layers[-1]):
                # check if output is correct
                if int(exp) == int(act):
                    continue

                neuron = network.neurons[uid]
                # apply delta learning to neuron's weights
                neuron.weights = [ w + learnfac * inval * (exp - act) \
                                    for w, inval in \
                                    zip(neuron.weights, sample[0]) ]

    def check_train_results():
        '''
        Tests all datasamples against trained network and checks if
        all results match the expected results.

        @return: True if all results match the expected ones, else False
        '''
        for sample in data:
            outvals = network.update(sample[0])

            assert(len(sample[1]) == len(outvals))

            # check for all output neurons if results
            # match the expected result
            for exp, act in zip(sample[1], outvals):
                if int(exp) != int(act):
                    return False

        return True

    while True:
        train_step()
        if check_train_results():
            break

def back_propagation(network, data, learnfac=0.5, min_err=0.3, max_cycles=10000):
    '''
    Trains the given network with back propagation rule. The parameter
    data has to have the following format:

    [
        (invals1, outvals1),
        (invals2, outvals2),
        (invals3, outvals3),
        ...
    ]

    @param network: the network to be trained
    @param data: the data which is used to train the network
    '''

    cycles = 0
    total_err = 0

    def train_step():
        '''
        A single training step that goes through all data samples
        and applies the back propagation rule.
        '''

        nonlocal cycles, total_err

        cycles += 1
        total_err = 0
        errsig = [ 0 for _ in network.neurons]

        for sample in data:
            outvals = network.update(sample[0])

            # go through all output neurons
            # and calculate their error
            for exp, act, uid in zip(sample[1], outvals, network.layers[-1]):
                err = exp - act
                errsig[uid] = act * (1 - act) * err
                total_err += math.pow(err, 2)

            # go through all remaining layers
            # backwards and calculate their error
            for l in range(len(network.layers) - 2, -1, -1):
                for uid1 in network.layers[l]:
                    errsum = 0
                    n1 = network.neurons[uid1]

                    # go through all neurons in next layer
                    # and calculate error from connected ones
                    for uid2 in network.layers[l + 1]:
                        n2 = network.neurons[uid2]
                        for i, inuid in enumerate(network.edges_ba[uid2]):
                            if inuid == uid1:
                                errsum += n2.weights[i] * errsig[uid2]

                    # calculate error signal
                    errsig[uid1] = n1.outval * (1 - n1.outval) * errsum

                    # go through all neurons in next layer
                    # and apply new weights
                    for uid2 in network.layers[l + 1]:
                        n2 = network.neurons[uid2]
                        for i, inuid in enumerate(network.edges_ba[uid2]):
                            if inuid == uid1:
                                n2.weights[i] = n2.weights[i] + learnfac * errsig[uid2] * n1.outval

    def check_train_results():
        return total_err <= min_err or cycles >= max_cycles

    while True:
        train_step()
        if check_train_results():
            break

