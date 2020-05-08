import torch
import torch.nn as nn

from LinearModel import *
outDim = 0
inDim = 1

def outStrengthCentralities(linearNetwork):

    for i in range(len(linearNetwork.layers)):
        layer = linearNetwork.layers[i].weight
        inFromPrevious = torch.sum(layer, dim=outDim) # suming along the outDim axis returns the shape of the inDIm
        if i == 0:
            strengths = inFromPrevious
        else:
            strengths = torch.cat((strengths, inFromPrevious))

        if i == len(linearNetwork.layers) - 1:
            noOutForLastLayer = torch.zeros(layer.shape[outDim])
            strengths = torch.cat((strengths, noOutForLastLayer), dim=0)

    return strengths

def inStrengthCentralities(linearNetwork):

    for i in range(len(linearNetwork.layers)):
        layer = linearNetwork.layers[i].weight

        if i == 0:
            # the first layer (the input signal) has 0 in-strength
            noInForFirstLayer = torch.zeros(layer.shape[inDim])
            strengths = noInForFirstLayer
            # strengths = torch.cat((strengths, noInForFirstLayer),dim=0)

        # oddly enough, you use the out-dimension from the current layer to calculate in-dimension for the next layer
        outFromPrevious = torch.sum(layer, dim=outDim)

        strengths = torch.cat((strengths, outFromPrevious))

    return strengths


# FIXME: right now this only calculate for positive 'strength'
def generateHistogramXY(tensor, plusOne=True):
    # get the range
    maxVal = torch.max(tensor)
    minVal = torch.tensor(0.0) #, requires_grad=True) #torch.min(tensor)
    valueRange = maxVal - minVal
    numSteps = 10

    stepSize =valueRange / float(numSteps)

    steps = torch.linspace(minVal.item(), maxVal.item(), numSteps, requires_grad=True)

    for i in range(steps.shape[0] - 1):
        localMinVal = steps[i]
        localMaxVal = steps[i + 1]

        nextX = torch.tensor([(localMinVal + localMaxVal) / 2], requires_grad=True)
        if plusOne:
            nextX[0] += 1

        if i == 0:
            histogramX = nextX
        else:
            histogramX = torch.cat((histogramX, nextX), dim=0)



        mins = torch.full((tensor.shape[0],), localMinVal.item(), requires_grad=True)
        maxs = torch.full((tensor.shape[0],), localMaxVal.item(), requires_grad=True)

        greater = tensor > mins
        lesser = tensor < maxs
        between = greater == lesser
        stepCount = torch.sum(between)

        proportion = torch.tensor([stepCount / float(tensor.shape[0])], requires_grad=True)

        proportion[0] += 1  # helps with the log
        if i == 0:
            histogramY = proportion
        else:
            histogramY = torch.cat((histogramY, proportion), dim=0)

        print(histogramX)
        print(histogramY)

    # histogramY = log(histogramY)
    # histogramX = log(histogramX)
    return histogramX, histogramY




class StructureLoss(nn.Module):

    ''' loss for optimizing a network to have statistics that match an ideal statistic '''

    def __init__(self, idealStrength):
        super(StructureLoss, self).__init__()
        self.idealStrength = torch.tensor(idealStrength)

    def calculateThresholds(self):
        # use torch.logspace() one for x values, one for y values
        pass

    def forward(self, model):

        inStrength = inStrengthCentralities(model)
        outStrength = outStrengthCentralities(model)

        totalStrength = inStrength + outStrength

        histX, histY = generateHistogramXY(totalStrength) # FIXME: this is where we loose the gradient
        loss = torch.sum(histY) # FIXME: remove this (I just placed it here for testing)
        #
        # histX = torch.log(histX)
        # histY = torch.log(histY)
        #
        # # expected values for each histX and y
        #
        # # we know a slope, and we'll use the first x point in histX
        # pointX = histX[0]
        # pointY = histY[0]
        #
        # # use y = mx + b
        # b = pointY - self.idealStrength * pointX
        #
        # loss = torch.tensor(0.0)
        #
        # for i in range(histX.shape[0]):
        #     x = histX[i]
        #     expectedVal = b + self.idealStrength * x
        #     actualVal = histY[i]
        #     loss += torch.abs(expectedVal - actualVal)
        #     i = i + 1

        return loss

print("started")
lossFunction = StructureLoss(0.1)
model = LinearModel(numLayers=5, inputSize=101, internalSize=101, outputSize=101, dropout=0)
optimizer = torch.optim.Adam(model.parameters())

numIters = 100
for i in range(numIters):
    optimizer.zero_grad()
    loss = lossFunction(model)
    loss.backward()
    optimizer.step()


    print(loss)

