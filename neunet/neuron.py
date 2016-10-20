'''
Created on 18 Oct 2016

@author: Fabian Meyer
'''

import math

class Neuron:

    def __init__(self, neuronId, activity, inWeights, activityThreshold):
        self._activityFuncs = {
            'euler' : self._calcEulerActivity,
            'jump'  : self._calcJumpActivity
        }

        assert(activity in self._activityFuncs)

        self._activity = activity
        self._id = neuronId
        self._inWeights = inWeights
        self._activityThreshold = activityThreshold
        self._outVal = 0.0

    def compute(self, inVals):
        netVal = self._calcNetVal(inVals)
        activity = self._activityFuncs[self._activity](netVal)
        self._outVal = activity

    def getOutValue(self):
        return self._outVal

    def _calcNetVal(self, inVals):
        assert(len(self._inWeights) == len(inVals))

        return sum([w * v for w, v in zip(self._inWeights, inVals)])

    def _calcEulerActivity(self, netVal):
        return 1 / (1 + math.exp(-self._activityThreshold * netVal))

    def _calcJumpActivity(self, netVal):
        if netVal >= self._activityThreshold:
            return 1
        else:
            return 0

class NeuralNetwork:

    def __init__(self, layers):
        self._nodes = []
        self._inEdges = []
        self._layers = [[] for _ in xrange(layers)]

    def addNeuron(self, layer, activity, inWeights, activityThreshold):
        assert(layer >= 0 and layer < len(self._layers))

        nid = len(self._nodes)
        self._nodes.append(Neuron(nid, activity, inWeights, activityThreshold))
        self._inEdges.append([])
        self._layers[layer].append(nid)

    def addEdge(self, neuronA, neuronB):
        assert(neuronA >= 0 and neuronA < len(self._nodes))
        assert(neuronB >= 0 and neuronB < len(self._nodes))

        self._inEdges[neuronB] = neuronA


    def compute(self, inValsPerNeuron):
        assert(self._layers)
        assert(len(inValsPerNeuron) == len(self._layers[0]))

        # first layer = input layer
        for inVals, neuID in zip(inValsPerNeuron, self._layers[0]):
            self._nodes[neuID].compute(inVals)

        # go through remaining _layers
        for layer in self._layers[1:]:
            # go through all _nodes in this layer
            for neuID in layer:
                inVals = [self._nodes[i]._outVal for i in self._inEdges[neuID]]
                self._nodes[neuID].compute(inVals)

        # return outVals of last layer = output layer
        return [self._nodes[nid]._outVal for nid in self._layers[-1]]
