'''
Created on 19 Oct 2016

@author: Fabian Meyer
'''

def binStrToVals(s):
    return [int(c) for c in s]

class Binary2UnaryTrainer:
    '''
    Trainer that implements delta learning for
    a binary to unary converter.
    '''

    def __init__(self, network, learnfac):
        self._network = network
        self._data = []
        self.learnfac = learnfac

    def load(self, fileName):
        print('Reading file "{0}" ...'.format(fileName))
        with open(fileName) as f:
            fcontent = f.readlines()

        print('Converting training data ...')
        self._data = []
        for l in fcontent:
            inVal, outVal = l.split()
            inVals = binStrToVals(inVal)
            inVals = [inVals for _ in self._network._layers[0]]
            outVals = binStrToVals(outVal)
            self._data.append((inVals, outVals))

    def _trainStep(self):
        for dat in self._data:
            outVals = self._network.compute(dat[0])

            assert(len(dat[1]) == len(outVals))
            assert(len(outVals) == len(self._network._layers[-1]))

            for exp, act, neuID in zip(dat[1], outVals, self._network._layers[-1]):
                if int(exp) == int(act):
                    continue

                neuron = self._network._nodes[neuID]
                neuron._inWeights = [w + self.learnfac * inval * (exp - act) for w, inval in zip(neuron._inWeights, dat[0][neuID])]

    def _testLearningProgress(self):
        for dat in self._data:
            outVals = self._network.compute(dat[0])

            assert(len(dat[1]) == len(outVals))

            for exp, act in zip(dat[1], outVals):
                if int(exp) != int(act):
                    return False

        return True

    def train(self):
        print('Training network ...')
        while True:
            self._trainStep()
            if self._testLearningProgress():
                break

        print('Done with training.')
        for n in self._network._nodes:
            print('-- id: {0}; weights: {1}'.format(n._id, n._inWeights))

class XorTrainer:
    '''
    Trainer that implements Backpropagation learning
    with Resilient Propagation.
    '''

    def __init__(self, network, learnfac):
        self._network = network
        self._data = []
        self.learnfac = learnfac

    def load(self, fileName):
        print('Reading file "{0}" ...'.format(fileName))
        with open(fileName) as f:
            fcontent = f.readlines()

        print('Converting training data ...')
        self._data = []
        for l in fcontent:
            inVal, outVal = l.split()
            inVals = binStrToVals(inVal)
            inVals = [[val] for val in inVals]
            assert(len(inVals) == len(self._network._layers[0]))

            outVals = binStrToVals(outVal)
            self._data.append((inVals, outVals))

    def train(self):
        print('Training network ...')

        errVals = [[dict() for _ in l] for l in self._network._layers]
        cycles = 0
        while cycles < 100:
            totalErr = 0
            cycles = cycles + 1

            for dat in self._data:
                outVals = self._network.compute(dat[0])

                assert(len(outVals) == len(self._network._layers))
                for n, exp, act in zip(self._network._layers[-1], dat[1], outVals):
                    errVals[-1][n]['o'] = act
                    errVals[-1][n]['f'] = exp - act
                    errVals[-1][n]['fsig'] = act * (1 - act) * errVals[0][n]['f']
                    totalErr = totalErr + pow(errVals[0][n]['f'], 2)

                # go through all remaining layers
                for l in range(len(self._network._layers) - 2, -1, -1):
                    for n1, err1 in zip(self._network._layers[l], errVals[l]):
                        fsum = 0
                        for n2, err2 in zip(self._network._layers[l + 1], errVals[l]):
                            inIdx = -1
                            for e in range(len(self._network._inEdges[n2])):
                                if self._network._inEdges[n2][e] == n2:
                                    inIdx = e
                                    break
                            if inIdx == -1:
                                continue

                            fsum = fsum + self._network._neurons[n2]._inWeights[inIdx] * err2['fsig']

                        err1['o'] = self._network._neurons[n1]._outVal
                        err1['fsig'] = err1['o'] * (1 - err1['o']) * fsum

                        for n2, err2 in zip(self._network._layers[l + 1], errVals[l]):
                            inIdx = -1
                            for e in range(len(self._network._inEdges[n2])):
                                if self._network._inEdges[n2][e] == n2:
                                    inIdx = e
                                    break
                            if inIdx == -1:
                                continue


