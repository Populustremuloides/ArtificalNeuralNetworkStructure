from AnalyzerScripts.Centralities import *
from AnalyzerScripts.Fractal import *
from AnalyzerScripts.PowerLaw import *
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

outputDir = "initialization_network_structures"
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

t1Path = "C:\\Users\\BCBrown\\PycharmProjects\\spectral_network\\Train\\initialization_test_output_neg5_to_3_noT-T1"

def removeNegative(strengthList):
    newList = []
    for item in strengthList:
        if item > 0:
            newList.append(item)
    return newList



def analyzeNetworks(folderPath):
    for fileName in os.listdir(folderPath):

        if fileName.endswith(".py"):
            pass
        elif fileName.endswith(".npy"):
            pass
        else:

                # open the file
                print(fileName)
                filePath = folderPath + "\\" + fileName
                model = LinearModel(numLayers=7, inputSize=101,internalSize=101,outputSize=101)
                model.load_state_dict(torch.load(filePath))

                # calculate degree centralities
                strength = strengthCentralities(model)
                np.save(outputDir + "\\strength_" + fileName, strength)

                G = getGraph(model)
                try:
                    btwn = betweenness(G)
                    np.save(outputDir + "\\betweenness_" + fileName, btwn)
                except:
                    print("unable to process betweenness centrality")

                try:
                    eig = eigen(G)
                    np.save(outputDir + "\\eigen_" + fileName, eig)
                except:
                    print("unable to process eigenvector centrality")

analyzeNetworks(t1Path)


#         strength = removeNegative(strength)
#         print(len(strength))
#         strength.sort()
#         absStrength.sort()
#
#         if "no_training" in fileName:
#             noT.append(negLogLogHistSlope(strength))
#             # noTAbs.append(negLogLogHistSlope(absStrength))
#         elif "0.0" in fileName:
#             do0.append(negLogLogHistSlope(strength))
#             # do0Abs.append(negLogLogHistSlope(absStrength))
#         elif "0.2" in fileName:
#             do2.append(negLogLogHistSlope(strength))
#             # do2Abs.append(negLogLogHistSlope(absStrength))
#         elif "0.4" in fileName:
#             do4.append(negLogLogHistSlope(strength))
#             # do4Abs.append(negLogLogHistSlope(absStrength))
#
#
# print(noT)
# print(do0)
# print(do2)
# print(do4)
#
# # because it is paired, I can only examine the pair-wise differences. . .
# def subtractElementWise(list1, list2):
#     assert len(list1) == len(list2)
#     differences = []
#     for i in range(len(list1)):
#         differences.append(list1[i] - list2[i])
#     return differences
#
# noT_do0 = subtractElementWise(noT, do0)
# do0_do2 = subtractElementWise(do0, do2)
# do2_do4 = subtractElementWise(do2, do4)
#
# print(noT_do0)
# print(do0_do2)
# print(do2_do4)
#
# fVal, pVal = stats.f_oneway(noT_do0, do0_do2, do2_do4)
# print('p value')
# print(pVal)
#
# diffDict = {}
# diffDict["difference"] = []
# diffDict["treatment"] = []
# for item in noT_do0:
#     diffDict["difference"].append(item)
#     diffDict["treatment"].append("no_training_to_train_do=0")
#
# for item in do0_do2:
#     diffDict["difference"].append(item)
#     diffDict["treatment"].append("do=0_to_do=0.2")
#
# for item in do2_do4:
#     diffDict["difference"].append(item)
#     diffDict["treatment"].append("do=2_to_do=0.4")
#
# df = pd.DataFrame.from_dict(diffDict)
# ax = sns.boxplot(x="treatment",y="difference", data=df)
# plt.savefig("differences")
# plt.show()
#
#
#
#
#
#
# def percentDiff(list1, list2):
#     assert len(list1) == len(list2)
#     percentDifferences = []
#     for i in range(len(list1)):
#         percentDifferences.append(abs(list1[i] - list2[i]) / abs(list1[i]))
#     return percentDifferences
#
# noT_do0P = percentDiff(noT, do0)
# do0_do2P = percentDiff(do0, do2)
# do2_do4P = percentDiff(do2, do4)
#
# percentDict = {}
# percentDict["percent_difference"] = []
# percentDict["treatment"] = []
# for item in noT_do0P:
#     percentDict["percent_difference"].append(item)
#     percentDict["treatment"].append("no_training_to_train_do=0")
#
# for item in do0_do2P:
#     percentDict["percent_difference"].append(item)
#     percentDict["treatment"].append("do=0_to_do=0.2")
#
# for item in do2_do4P:
#     percentDict["percent_difference"].append(item)
#     percentDict["treatment"].append("do=2_to_do=0.4")
#
# df = pd.DataFrame.from_dict(percentDict)
# ax = sns.boxplot(x="treatment",y="percent_difference", data=df)
# plt.savefig("percent_differences")
# plt.show()

