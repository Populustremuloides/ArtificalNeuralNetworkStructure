from Models.LinearModel import *
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import random

# hyperparameters
lr = 1e-2
epochs = 20
batchSize = 20
cuda = False
lossFunction = nn.L1Loss()
validationInterval = 100

# load the training dataset
trainDataPath = "C:\\Users\\BCBrown\\PycharmProjects\\spectral_network\\DataGenerationScripts\\neg5to3\\train_no_transformation_sequences.npy"

trainData = np.load(trainDataPath)
print(len(trainData))

# load the target dataset
trainT1DataPath = "C:\\Users\\BCBrown\\PycharmProjects\\spectral_network\\DataGenerationScripts\\neg5to3\\train_t1_accentuate_small_sequences.npy"
trainT1Data = np.load(trainT1DataPath)
print(len(trainT1Data))

# load the testing dataset
testDataPath = "C:\\Users\\BCBrown\\PycharmProjects\\spectral_network\\DataGenerationScripts\\neg5to3\\test_no_transformation_sequences.npy"
testData = np.load(testDataPath)
print(len(testData))

testT1DataPath = "C:\\Users\\BCBrown\\PycharmProjects\\spectral_network\\DataGenerationScripts\\neg5to3\\test_t1_accentuate_small_sequences.npy"
testT1Data = np.load(testT1DataPath)
print(len(testT1Data))


def trainModel(model):
    # calculated values:
    numBatches = trainData.shape[0] // batchSize

    # other stuff
    optimizer = optim.Adam(model.parameters())
    trainLosses = []

    trainIndices = list(range(trainData.shape[0]))
    for epoch in range(epochs):
        random.shuffle(trainIndices)

        loop = tqdm(total=len(trainIndices), position=0, leave=False)
        for batch in range(numBatches):
            model.train()
            optimizer.zero_grad()

            startIndex = batch * batchSize
            endIndex = startIndex + batchSize
            indices = trainIndices[startIndex:endIndex]

            trainBatch = []
            targetBatch = []
            for index in indices:
                trainBatch.append(trainData[index])
                targetBatch.append(trainT1Data[index])

            trainBatch = torch.Tensor(trainBatch)
            targetBatch = torch.Tensor(targetBatch)

            out = model(trainBatch)

            loss = lossFunction(out, targetBatch)
            trainLosses.append(loss.item())

            loss.backward()
            optimizer.step()

            loop.set_description(
                    'epoch:{}, loss:{:,.4f}'.format(
                        epoch, loss))
            loop.update(batchSize)

    model.eval()
    finalLosses = []
    for i in range(len(testData)):
        sourceBatch = torch.tensor([testData[i]])
        targetBatch = torch.tensor([testT1Data[i]])
        outBatch = model(sourceBatch)
        loss = lossFunction(outBatch, targetBatch)
        finalLosses.append(loss.item())

    return model, trainLosses, finalLosses

from torch.nn.init import *
import os
def weights_init_uniform(m):
    if isinstance(m, nn.Linear):
        uniform_(m.weight.data)
        zeros_(m.bias.data)

def weights_init_orthogonal(m):
    if isinstance(m, nn.Linear):
        orthogonal_(m.weight.data)
        zeros_(m.bias.data)

def weights_init_kaiming_normal(m):
    if isinstance(m, nn.Linear):
        kaiming_normal_(m.weight.data)
        zeros_(m.bias.data)
def weights_init_kaiming_uniform(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)
        zeros_(m.bias.data)
def weights_init_xe_uniform(m):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight.data)
        zeros_(m.bias.data)
def weights_init_xe_normal(m):
    if isinstance(m, nn.Linear):
        xavier_normal_(m.weight.data)
        zeros_(m.bias.data)

#model.apply(weights_init)

outPath = "initialization_test_output_neg5_to_3_noT-T1"
if not os.path.exists(outPath):
    os.mkdir(outPath)

for test in range(50):

    print()

    model_0 = LinearModel(numLayers=7, inputSize=101, internalSize=101, outputSize=101, dropout=0.0)
    model_2 = LinearModel(numLayers=7, inputSize=101, internalSize=101, outputSize=101, dropout=0.2)
    model_4 = LinearModel(numLayers=7, inputSize=101, internalSize=101, outputSize=101, dropout=0.4)

    if test % 5 == 2:
        initializationType = "kaiming_normal"
        model_0.apply(weights_init_kaiming_normal)
        model_2.apply(weights_init_kaiming_normal)
        model_4.apply(weights_init_kaiming_normal)
    elif test % 5 == 1:
        initializationType = "uniform"
        model_0.apply(weights_init_uniform)
        model_2.apply(weights_init_uniform)
        model_4.apply(weights_init_uniform)
    elif test % 5 == 3:
        initializationType = "kaiming_uniform"
        model_0.apply(weights_init_kaiming_uniform)
        model_2.apply(weights_init_kaiming_uniform)
        model_4.apply(weights_init_kaiming_uniform)
    elif test % 5 == 4:
        initializationType = "xe_uniform"
        model_0.apply(weights_init_xe_uniform)
        model_2.apply(weights_init_xe_uniform)
        model_4.apply(weights_init_xe_uniform)
    elif test % 5 == 5:
        initializationType = "xe_normal"
        model_0.apply(weights_init_xe_normal)
        model_2.apply(weights_init_xe_normal)
        model_4.apply(weights_init_xe_normal)
    elif test % 5 == 0:
        initializationType = "orthogonal"
        model_0.apply(weights_init_orthogonal)
        model_2.apply(weights_init_orthogonal)
        model_4.apply(weights_init_orthogonal)

    prefix = outPath + "\\noT-T1_" + str(test) + "_" + initializationType + "_"
    print("repeat number " + str(test) + "**********************")

    torch.save(model_0.state_dict(), prefix + "_model_0-0_dropout_untrained") # save all of the untrained
    torch.save(model_2.state_dict(), prefix + "_model_0-2_dropout_untrained") # save all of the untrained
    torch.save(model_4.state_dict(), prefix + "_model_0-4_dropout_untrained") # save all of the untrained

    print('no dropout')
    model_0, loss_0, test_losses_0 = trainModel(model_0)
    torch.save(model_0.state_dict(), prefix + "_model_0-0_dropout_trained")
    np.save(prefix + "training_0-0_dropout_loss", loss_0)
    np.save(prefix + "final_0-0_dropout_test_loss", test_losses_0)
    print()

    print('0.2 dropout')
    model_2, loss_2, test_losses_2 = trainModel(model_2)
    torch.save(model_2.state_dict(), prefix + "_model_0-2_dropout_trained")
    np.save(prefix + "training_0-2_dropout_loss",loss_2)
    np.save(prefix + "final_0-2_dropout_test_loss", test_losses_2)
    print()

    print('0.4 dropout')
    model_4, loss_4, test_losses_4 = trainModel(model_4)
    torch.save(model_4.state_dict(), prefix + "_model_0-4_dropout_trained")
    np.save(prefix + "training_0-4_dropout_loss", loss_4)
    np.save(prefix + "final_0-4_dropout_test_loss", test_losses_4)
    print()

