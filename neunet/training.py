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

    def __init__(self, network, learnfac, maxerr):
        self._network = network
        self._data = []
        self.learnfac = learnfac
        self.maxerr = maxerr

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
        totalErr = self.maxerr * 2
        # train until x cycles is reached
        while totalErr > self.maxerr and cycles < 1000:
            totalErr = 0
            cycles = cycles + 1

            for dat in self._data:
                outVals = self._network.compute(dat[0])

                # loop first through output layer
                assert(len(outVals) == len(errVals[-1]))
                for exp, act, err1 in zip(dat[1], outVals, errVals[-1]):
                    err = exp - act
                    errsig = act * (1 - act) * err
                    err1['o'] = act
                    err1['err'] = err
                    err1['errsig'] = errsig
                    totalErr = totalErr + pow(err, 2)

                # go through all remaining layers
                for lidx in range(len(self._network._layers) - 2, -1, -1):
                    lcurr = self._network._layers[lidx]
                    lnext = self._network._layers[lidx + 1]
                    errCurr = errVals[lidx]
                    errNext = errVals[lidx + 1]

                    # go through all neurons in current layer
                    for n1, err1 in zip(lcurr, errCurr):
                        errsum = 0

                        # go through all neurons in next layer
                        for n2, err2 in zip(lnext, errNext):
                            neuron2 = self._network._nodes[n2]
                            inEdges2 = self._network._inEdges[n2]

                            # find weight that maps to n1
                            for inE, w in zip(inEdges2, neuron2._inWeights):
                                if inE == n1:
                                    errsum = errsum + w * err2['errsig']

                        err1['o'] = self._network._nodes[n1]._outVal
                        err1['errsig'] = err1['o'] * (1 - err1['o']) * errsum

                        for n2, err2 in zip(lnext, errNext):
                            neuron2 = self._network._nodes[n2]
                            inEdges2 = self._network._inEdges[n2]

                            # find weight that maps to n1
                            for widx in range(len(neuron2._inWeights)):
                                if inEdges2[widx] == n1:
                                    neuron2._inWeights[widx] = neuron2._inWeights[widx] + self.learnfac * err1['o'] * err2['errsig']
            print('-- totalerr: {0}; maxerr: {1}'.format(totalErr, self.maxerr))
