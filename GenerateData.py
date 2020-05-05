import random
from AnalyzerScripts.Fractal import *
from AnalyzerScripts.PowerLaw import *
from DataGenerationScripts.Transformations import *
import pandas as pd
import seaborn as sns
import os


print("started")
# will this experiment work differently if I don't use a gradient of alphas? i.e. a dataset that is totally one alpha? I can try both.

# dataset will include the following per tuple:
# - 1d array
# - 2d copy of array
# - 2d fourier decomposition
# - 1d magnitude fourier decomposition
# - fractal dimension
# - spectral slope
# - transformed versions?
# it might be faster to make several different datasets

# generate the dataset by adjusting alpha in the following equation:
# D(f) = 1/f^alpha

# generate another dataset that is the same thing but slightly noisy
# D(f) = 1/f^(alpha + np.random.sample() * 1e-3)

# generate another dataset that is noisy and not all less than 1

# hyperparameters
minAlpha = 0
maxAlpha = 3
numTrainDivisions = 10000 # number of divisions between min and max alpha
numTestDivisions = 1000
sequenceLength = 101 # length of data (must be an odd number because of hacky fourier details)
theta = math.pi / 4 # radians between frequency magnitude and real density (between 0 and math.pi /2)
# theta = 0 implies only real values
# theta = math.pi / 2 implies only imaginary values
jitterFactor = math.pi/2

outputDir = "zeroto3"

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

def generateSequence(alpha, noisy=True):
    densities = []
    for f in range(1, (sequenceLength // 2) + 2):
        densityMagnitude = 1 / (f**alpha)

        if noisy:
            r1 = random.random() * jitterFactor
            r2 = random.random() * jitterFactor
            rI = random.randint(1,10)
            if rI % 2 == 0:
                r1 = r1 * -1
            rI = random.randint(1,10)
            if rI % 2 == 0:
                r2 = r2 * -1
        else:
            r1 = 0
            r2 = 0
        # definition: densityMagnitude = sqrt(realDensity^2 + imaginaryDensity^2)
        realDensity = math.cos(theta + r1) * densityMagnitude
        imaginaryDensity = math.sin(theta + r2) * densityMagnitude

        density = [realDensity, imaginaryDensity]
        densities.append(density)

    densities = torch.FloatTensor(densities)
    values = torch.irfft(densities, signal_ndim=1)
    values = values - torch.mean(values) # mean 0
    values = values.numpy()

    return values



def curveToArray(curve):
    dilationFactor = 10
    # scale the values of curve so that the maxVal = len(curve), minval = 0

    curve = curve + np.abs(np.min(curve)) # set the minimum value to 0
    curveMax = np.max(curve)
    curveMin = 0

    curveRange = curveMax - curveMin # added here just for clarity
    scalingFactor = len(curve) / curveRange # * dilationFactor) - dilationFactor

    curve = curve * scalingFactor

    z = np.zeros((len(curve), len(curve) * dilationFactor - dilationFactor)) # -dilation factor because we are interpolating
    # print(z.shape)
    ys = []
    for i in range(len(curve) - 1):
        baseIndex = i * dilationFactor
        for d in range(dilationFactor):
            # get the x value
            index = baseIndex + d

            # get the y value

            # calculate the slope between this point and the next
            x1 = i * dilationFactor
            y1 = curve[i]

            x2 = (i + 1) * dilationFactor
            y2 = curve[(i + 1)]

            slope = (y2 - y1) / (x2 - x1)

            # linearly interpolate between points
            y = y1 + (d * slope) - 1 # -1 because we zero index

            ys.append(int(y))

    x = 0
    for y in ys:
        z[y, x] = 1
        x = x + 1
    # plt.imshow(z)
    # plt.show()
    return z



def generateData(numDivisions, test=False):

    if test == True:
        prefix = outputDir + "\\test_"
    else:
        prefix = outputDir + "\\train_"

    # calculated values
    currentAlpha = minAlpha
    alphaRange = maxAlpha - minAlpha
    increment = alphaRange / (numDivisions - 1)  # -1, to end on the max value

    # data storage
    noTSeqs = []
    t1Seqs = []
    t2Seqs = []
    t3Seqs = []
    t4Seqs = []
    t5Seqs = []

    longdfdict = {}
    longdfdict["transformation"] = []
    longdfdict["start_alpha"] = []
    longdfdict["end_alpha"] = []
    longdfdict["start_fractal_dimension"] = []
    longdfdict["end_fractal_dimension"] = []

    widedfdict = {}
    widedfdict["index"] = []
    widedfdict["no_t_alpha"] = []
    widedfdict["t1_alpha"] = []
    widedfdict["t2_alpha"] = []
    widedfdict["t3_alpha"] = []
    widedfdict["t4_alpha"] = []
    widedfdict["t5_alpha"] = []
    widedfdict["no_t_ds"] = []
    widedfdict["t1_ds"] = []
    widedfdict["t2_ds"] = []
    widedfdict["t3_ds"] = []
    widedfdict["t4_ds"] = []
    widedfdict["t5_ds"] = []

    for i in range(numDivisions):

        # generate sequence and analyze it
        sequence = normalize(generateSequence(currentAlpha, noisy=True), 1, -1)
        z = curveToArray(sequence)
        noTAlpha = negLogLogSpectralSlope(sequence)
        d = fractal_dimension(z, threshold=0.1)

        longdfdict["transformation"].append("none")
        longdfdict["start_alpha"].append(currentAlpha)
        longdfdict["end_alpha"].append(noTAlpha)
        longdfdict["start_fractal_dimension"].append(d)
        longdfdict["end_fractal_dimension"].append(d)

        noTSeqs.append(sequence)

        # transform sequence and analyze it

        # T1: accentuate small
        t1Sequence = normalize(accentuateSmall(normalize(sequence, 1.1, 2), 2), 1, -1)
        t1z = curveToArray(t1Sequence)
        t1Alpha = negLogLogSpectralSlope(t1Sequence)
        t1d = fractal_dimension(t1z, threshold=0.1)
        longdfdict["transformation"].append("accentuate_small")
        longdfdict["start_alpha"].append(currentAlpha)
        longdfdict["end_alpha"].append(t1Alpha)
        longdfdict["start_fractal_dimension"].append(d)
        longdfdict["end_fractal_dimension"].append(t1d)

        t1Seqs.append(t1Sequence)

        # T2: accentuate large
        t2Sequence = normalize(accentuateLarge(normalize(sequence, 1.1, 2), 2), 1, -1)
        t2z = curveToArray(t2Sequence)
        t2Alpha = negLogLogSpectralSlope(t2Sequence)
        t2d = fractal_dimension(t2z, threshold=0.1)
        longdfdict["transformation"].append("accentuate_large")
        longdfdict["start_alpha"].append(currentAlpha)
        longdfdict["end_alpha"].append(t2Alpha)
        longdfdict["start_fractal_dimension"].append(d)
        longdfdict["end_fractal_dimension"].append(t2d)

        t2Seqs.append(t2Sequence)

        # T3: blockify
        t3Sequence = normalize(blockify(normalize(sequence, 1.1, 2), 5), 1, -1)
        t3z = curveToArray(t3Sequence)
        t3Alpha = negLogLogSpectralSlope(t3Sequence)
        t3d = fractal_dimension(t3z, threshold=0.1)
        longdfdict["transformation"].append("blockify")
        longdfdict["start_alpha"].append(currentAlpha)
        longdfdict["end_alpha"].append(t3Alpha)
        longdfdict["start_fractal_dimension"].append(d)
        longdfdict["end_fractal_dimension"].append(t3d)

        t3Seqs.append(t3Sequence)

        # T4: reverse
        t4Sequence = normalize(reverse(normalize(sequence, 1.1, 2)), 1, -1)
        t4z = curveToArray(t4Sequence)
        t4Alpha = negLogLogSpectralSlope(t4Sequence)
        t4d = fractal_dimension(t4z, threshold=0.1)
        longdfdict["transformation"].append("reverse")
        longdfdict["start_alpha"].append(currentAlpha)
        longdfdict["end_alpha"].append(t4Alpha)
        longdfdict["start_fractal_dimension"].append(d)
        longdfdict["end_fractal_dimension"].append(t4d)

        t4Seqs.append(t4Sequence)

        # T5: translateX
        t5Sequence = normalize(translateX(normalize(sequence, 1.1, 2), len(sequence) // 2), 1, -1)
        t5z = curveToArray(t5Sequence)
        t5Alpha = negLogLogSpectralSlope(t5Sequence)
        t5d = fractal_dimension(t5z, threshold=0.1)
        longdfdict["transformation"].append("translateX")
        longdfdict["start_alpha"].append(currentAlpha)
        longdfdict["end_alpha"].append(t5Alpha)
        longdfdict["start_fractal_dimension"].append(d)
        longdfdict["end_fractal_dimension"].append(t5d)

        t5Seqs.append(t5Sequence)

        # update the wide df
        widedfdict["index"].append(i)
        widedfdict["no_t_alpha"].append(noTAlpha)
        widedfdict["t1_alpha"].append(t1Alpha)
        widedfdict["t2_alpha"].append(t2Alpha)
        widedfdict["t3_alpha"].append(t3Alpha)
        widedfdict["t4_alpha"].append(t4Alpha)
        widedfdict["t5_alpha"].append(t5Alpha)

        widedfdict["no_t_ds"].append(d)
        widedfdict["t1_ds"].append(t1d)

        widedfdict["t2_ds"].append(t2d)

        widedfdict["t3_ds"].append(t3d)

        widedfdict["t4_ds"].append(t4d)

        widedfdict["t5_ds"].append(t5d)
        # increment
        currentAlpha = currentAlpha + increment

        if i % 10 == 0:
            print(i)

    df = pd.DataFrame.from_dict(longdfdict)

    sns.lineplot(x="start_alpha", y="end_alpha", hue="transformation", data =df)
    plt.savefig(prefix + "startVendAlpha.png")
    plt.clf()

    sns.lineplot(x="start_fractal_dimension", y="end_fractal_dimension", hue="transformation",data=df)
    plt.savefig(prefix + "startVendFractal.png")
    plt.clf()

    sns.lineplot(x="start_alpha", y="start_fractal_dimension", hue="transformation",data=df)
    plt.savefig(prefix + "startAlphaVstartFractal.png")
    plt.clf()

    sns.lineplot(x="end_alpha", y="end_fractal_dimension", hue="transformation",data=df)
    plt.savefig(prefix + "endAlphaVstartFractal.png")
    plt.clf()

    widedf = pd.DataFrame.from_dict(widedfdict)
    widedf.to_csv(prefix + "modulated_fractal_data_wide.csv")

    df.to_csv(prefix + "modulated_fractal_data_long.csv")

    np.save(prefix + "no_transformation_sequences", noTSeqs)
    np.save(prefix + "t1_accentuate_small_sequences", t1Seqs)
    np.save(prefix + "t2_accentuate_large_sequences", t2Seqs)
    np.save(prefix + "t3_blockify_sequences", t3Seqs)
    np.save(prefix + "t4_reverse_sequences", t4Seqs)
    np.save(prefix + "t5_translate_x_sequences", t5Seqs)


print("generating training code ********** ")
generateData(numTrainDivisions, test=False)
print("generating testing code *********** ")
generateData(numTestDivisions, test=True)
