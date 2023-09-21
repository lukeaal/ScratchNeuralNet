import numpy as np


class MLP:
    inputsData = []
    targetsData = []
    # weights connecting input to hidden - size if #I * #H
    vWeights = []
    # weigths connecting hidden to output - size of #H * #O
    wWeights = []
    # rho is learning rate
    rho = 0
    # current out put activation after forward pass, each item is a function of softmax
    # maximum item's index is the prediction for that pass.
    currInputActivation = []
    currOutputActivation = []
    curOutputTarget = []
    # current hidden layer activation, all values here are fed through sigmoid
    # sums of the weights multiplied by the input
    currHiddenActivation = []

    # number of hidden nodes, inputs, targets, learning rate, itterations
    def __init__(self, inputs, targets, numHiddenNode, rho) -> None:
        # set the dimensions of the weights and initilize their value randomly
        self.rho = rho
        self.inputsData = inputs
        self.targetsData = targets
        # create randomly good starting value for the weights.
        # small numbers positive and negative around zero
        self.vWeights = np.random.uniform(
            low=-.02, high=.02, size=(numHiddenNode, len(inputs[0])))
        self.wWeights = np.random.uniform(
            low=-.02, high=.02, size=(len(targets[0]), numHiddenNode))
        self.currHiddenActivation = np.zeros(numHiddenNode)
        self.currOutputActivation = np.zeros(len(targets[0]))
        self.currOutPutErr = np.zeros(len(targets[0]))
        self.currHiddenErr = np.zeros(numHiddenNode)
        pass

    # sigmoid function for hidden layer Activation
    def sigmoid(self, nums) -> float:
        return [1/(1 + np.exp(-num)) for num in nums]
    # softmax fuction for output node Activation

    def softMax(self, nums) -> list[int]:
        return np.exp(nums) / np.sum(np.exp(nums), axis=0)

    def forwardPass(self, inputLayer) -> list[int]:
        self.currInputActivation = inputLayer
        hiddenSums = []
        outputSums = []
        # use inputLayer and v weights to find hidden nodes activation
        for i in range(len(self.currHiddenActivation)):
            # sum the weights times the input
            hiddenSums.append(
                sum(np.array(inputLayer)*np.array(self.vWeights[i])))
        # use hidden node activation and w weights to find output activation
        self.currHiddenActivation = self.sigmoid(hiddenSums)
        for i in range(len(self.wWeights)):
            outputSums.append(
                sum(np.array(self.currHiddenActivation)*np.array(self.wWeights[i])))
        # update current activations
        self.currOutputActivation = self.softMax(outputSums)

        # return the current
        return self.currOutputActivation

    currOutPutErr = []
    currHiddenErr = []
    """
    only run this after you have done one forward pass.
    """

    def backwardsPass(self, groundTruth):
        self.curOutputTarget = groundTruth

        # first, find the error of output layer.
        for i in range(len(self.currOutPutErr)):
            self.currOutPutErr[i] = (self.currOutputActivation[i] - self.curOutputTarget[i])*(
                self.currOutputActivation[i])*(1-self.currOutputActivation[i])

        # second, compute the error of the hidden layer.
        for x in range(len(self.currHiddenErr)):
            sigma = 0
            for j in range(len(self.currOutPutErr)):
                sigma += self.wWeights[j][x]*self.currOutPutErr[j]
            self.currHiddenErr[x] = (
                self.currHiddenActivation[x])*(1-self.currHiddenActivation[x])*sigma

        # third, update the output layer weights (wWeights)
        for u in range(len(self.wWeights)):
            for g in range(len(self.wWeights[0])):
                self.wWeights[u][g] = self.wWeights[u][g] - \
                    (self.rho)*(self.currOutPutErr[u]
                                )*(self.currHiddenActivation[g])

        # fourth, update the hidden layer weights (vWeights)
        for h in range(len(self.vWeights)):
            for l in range(len(self.currInputActivation)):
                self.vWeights[h][l] = self.vWeights[h][l] - \
                    (self.rho)*(self.currHiddenErr[h]
                                )*(self.currInputActivation[l])
        return self.vWeights

    """
    at this point, the data is inside the class, the weights have been 
    initilized, rho and number of hidden nodes has been determined.
    All this does is itterate over the current data, running a forward 
    pass and then a backwardpass.
    """

    def train(self):
        for i in range(len(self.inputsData)):
            self.forwardPass(self.inputsData[i])
            self.backwardsPass(self.targetsData[i])
        pass
    # sum of square errors for testing acuracy

    def sse(self):
        sse = 0
        for data in range(len(self.targetsData)):
            result = self.forwardPass(self.inputsData[data])
            actual = self.targetsData[data]
            for num in range(len(result)):
                dif = result[num] - actual[num]
                sse += dif*dif
        return sse

        # takes in the resuilt fro ma forward pass, and returns
        # the index of the max

    def guess(self, inputToGuess):
        return list(inputToGuess).index(max(inputToGuess))
    # find accuracy of model on unseen data

    def accuracy(self, valInputs, valTargets):
        correct = 0
        for i in range(len(valInputs)):
            guess = self.forwardPass(valInputs[i])
            if self.guess(valTargets[i]) == self.guess(guess):
                correct += 1
        return correct/len(valInputs)
