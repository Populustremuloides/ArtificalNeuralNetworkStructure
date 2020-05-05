import torch
import torch.nn as nn
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

class LinearModel(nn.Module):
    def __init__(self, numLayers, inputSize, internalSize, outputSize, dropout=0, init="o"):
        super(LinearModel, self).__init__()
        self.layers = nn.ModuleList([])
        self.inputSize = inputSize

        for i in range(numLayers):
            if i == 0:
                newLayer = nn.Linear(in_features=inputSize, out_features=internalSize)
            elif i == numLayers - 1:
                newLayer = nn.Linear(in_features=inputSize, out_features=outputSize)
            else:
                newLayer = nn.Linear(in_features=internalSize, out_features=internalSize)

            self.layers.append(newLayer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batchSize = x.shape[0]

        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            x = self.dropout(self.relu(layer(x)))

        return self.layers[-1](x)



