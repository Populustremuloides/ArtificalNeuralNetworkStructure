import math
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def log(array):
    array = np.asarray(array, dtype="float32")
    for i in range(array.shape[0]):
        if array[i] > 0:
            # print("lESS THAN 0: " + str(array[i]))
            # array[i] = 0.0000000000001
            array[i] = math.log(array[i], math.e)
    return array

def translateSquare(array, translation, power):
    array = np.asarray(array, dtype="float32")
    for i in range(array.shape[0]):
        array[i] = array[i] - translation
        array[i] = math.pow(array[i], power)
    return array


def negLogLogSpectralSlope(array):
    # fourier decomposiiton
    tensor = torch.FloatTensor(array)
    spectralPowers = torch.rfft(tensor, signal_ndim=1)
    spectralPowers = spectralPowers.numpy()

    # convert into magnitude
    magnitudes = []
    for i in range(spectralPowers.shape[0]):
        real = spectralPowers[i,0]
        imaginary = spectralPowers[i,1]
        magnitude = math.sqrt(real**2 + imaginary**2)
        magnitudes.append(magnitude)

    # log it
    magnitudes = log(magnitudes)

    # x axis
    x = list(range(1, len(magnitudes) + 1))
    # log it
    x = log(x)

    # linear regression
    reg = LinearRegression().fit(np.asarray(x).reshape(-1,1),np.asarray(magnitudes))
    slope = reg.coef_[0]
    return -slope


def generateHistogramXY(array, plusOne=True):
    # get the range
    maxVal = max(array)
    valueRange = maxVal
    numSteps = 10
    stepSize = valueRange / numSteps

    currentVal = 0
    histogramX = []
    histogramY = []
    for step in range(numSteps):
        localMinVal = currentVal
        localMaxVal = currentVal + stepSize
        if plusOne:
            histogramX.append(((localMinVal + localMaxVal) / 2) + 1)
        else:
            histogramX.append(((localMinVal + localMaxVal) / 2))

        stepCount = 0
        for value in array:
            if value >= localMinVal and value < localMaxVal:
                stepCount += 1
        proportion = stepCount / len(array)
        proportion += 1  # helps with the log
        histogramY.append(proportion)

        currentVal += stepSize
    # histogramY = log(histogramY)
    # histogramX = log(histogramX)
    return histogramY, histogramX


def negLogLogHistSpectralSlope(array):
    histogramY,histogramX = generateHistogramXY(array)
    return negLogLogSpectralSlope(histogramY)

def negLogLogHistSlope(array):
    histogramY,histogramX = generateHistogramXY(array)
    histogramY = log(histogramY)
    histogramX = log(histogramX)
    reg = LinearRegression().fit(np.asarray(histogramX).reshape(-1,1),np.asarray(histogramY))
    slope = reg.coef_[0]

    return -slope

def translateSquareSlope(array):
    y, x = generateHistogramXY(array, plusOne=False)
    x = translateSquare(x, 0.4, 2)
    y = translateSquare(y, 0.4, 2)
    reg = LinearRegression().fit(np.asarray(x).reshape(-1,1),np.asarray(y))
    slope = reg.coef_[0]
    return slope

