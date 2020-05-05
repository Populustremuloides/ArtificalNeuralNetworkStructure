import torch
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from Models.LinearModel import *
import time
from Train.T3Results import *
import random

def getNodesAndEdges(linearModel):
    i = 0
    numNodes = 0
    numEdges = 0
    for layer in linearModel.layers:
        inNodes = layer.weight.shape[1]
        outNodes = layer.weight.shape[0]

        if i == 0:
            numNodes += inNodes
        numNodes += outNodes

        numEdges += inNodes * outNodes
        i = i + 1
    return numNodes, numEdges


outDim = 0
inDim = 1

def outStrengthCentralities(linearNetwork):
    strengths = torch.Tensor([])

    for i in range(len(linearNetwork.layers)):
        layer = linearNetwork.layers[i].weight
        inFromPrevious = torch.sum(layer, dim=outDim) # suming along the outDim axis returns the shape of the inDIm
        strengths = torch.cat((strengths, inFromPrevious))

        if i == len(linearNetwork.layers) - 1:
            noOutForLastLayer = torch.zeros(layer.shape[outDim])
            strengths = torch.cat((strengths, noOutForLastLayer), dim=0)

    return strengths.detach().numpy()

def inStrengthCentralities(linearNetwork):
    strengths = torch.Tensor([])
    for i in range(len(linearNetwork.layers)):
        layer = linearNetwork.layers[i].weight

        if i == 0:
            # the first layer (the input signal) has 0 in-strength
            noInForFirstLayer = torch.zeros(layer.shape[inDim])
            strengths = torch.cat((strengths, noInForFirstLayer),dim=0)

        # oddly enough, you use the out-dimension from the current layer to calculate in-dimension for the next layer
        outFromPrevious = torch.sum(layer, dim=outDim)

        strengths = torch.cat((strengths, outFromPrevious))

    return strengths.detach().numpy()


def strengthCentralities(linearNetwork):
    inStrength = inStrengthCentralities(linearNetwork)
    outStrength = outStrengthCentralities(linearNetwork)

    return inStrength + outStrength


def getGraph(linearNetwork, directed=False):
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    numPrevious = 0
    for layer in linearNetwork.layers:
        inNodes = layer.weight.shape[1]
        outNodes = layer.weight.shape[0]
        # go through every column
        for n in range(inNodes):
            nStrength = 0
            nAbsStrength = 0
            for m in range(outNodes):
                value = layer.weight[m,n]
                nAbsStrength += torch.abs(value).item()
                nStrength += value.item()

                nodeM = m + numPrevious + inNodes
                nodeN = n + numPrevious

                # treating the network like it is undirected
                G.add_edge(nodeN, nodeM, weight=value.item())
                if directed:
                    G.add_edge(nodeM, nodeN, weight=value.item())
        numPrevious += inNodes
    return G



def betweenness(G):
    return list(nx.betweenness_centrality(G, weight="weight").values())
def eigen(G):
    return list(nx.eigenvector_centrality_numpy(G, weight="weight").values())
def communities(G):
    return greedy_modularity_communities(G, weight="weight")


def percolation(G, probability):
    range = 10
    nodes = list(G.nodes)

    threshold = probability * range
    ranNum = random.randrange(1,range)

    percolationStates = []
    for node in nodes:
        value = 1
        if ranNum <= threshold:
            value = 0
        percolationStates.append(value)

    nodesToStates = dict(zip(nodes, percolationStates))

    nx.set_node_attributes(G, nodesToStates, "percolation")
    return nx.percolation_centrality(G, weight="weight")
