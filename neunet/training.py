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

