import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import int_shape

#Generate a simple sequence model taking one-hot encoded input and producing a logits profile and a log(counts) scalar. 

def _biasModelHead(dilateOutput, individualHead, outputFilterWidth):
    numOutputs = len(individualHead["data"])
    profile = keras.layers.Conv1D(numOutputs, outputFilterWidth, padding='valid', name='biasProfile_{0:s}'.format(individualHead["head-name"]))(dilateOutput)
    countsGap = keras.layers.GlobalAveragePooling1D(name='biasCountsGap_{0:s}'.format(individualHead["head-name"]))(dilateOutput)
    counts = keras.layers.Dense(numOutputs, name='biasLogcounts_{0:s}'.format(individualHead["head-name"]))(countsGap)
    #tf.print(profile)
    #tf.print(counts)
    return (profile, counts)



def biasModel(inputLength, outputLength, numFilters, numLayers, inputFilterWidth, outputFilterWidth, headList):
    """This is the classic BPNet architecture.
    inputLength is the length of the one-hot encoded DNA sequence. 
    outputLength is the width of the predicted profile. 
    numFilters is the number of convolutional filters used at each layer. 
    numLayers is the number of dilated convolutions. 
    inputFilterWidth is the width of the first convolutional layer, the one looking for motifs. 
    outputFilterWidth is the width of the profile head convolutional filter at the very bottom of the network.
    taskInfo is taken directly from a <bigwig-list> in the configuration JSON.

    Returns the TF model where: 
    Input to this model is a (batch x inputLength x 4) tensor of one-hot encoded DNA.
    Output is a list of (profilePreds, countPreds)
    profilePreds is a tensor of shape (batch x numTasks x outputLength), containing the 
    logits of the profile values for each task. 
    countsPreds is a tensor of shape (batch x numTasks) containing the log counts for each task. 

    It is an error to call this function with an inconsistent network structure, such as an input that is too long.
    """

    inputLayer = keras.Input((inputLength, 4), name='bias_input')

    initialConv = keras.layers.Conv1D(numFilters, kernel_size=inputFilterWidth, padding='valid',
            activation='relu', name="bias_initial_conv")(inputLayer)
    prevLayer = initialConv
    for i in range(numLayers):
        newConv = keras.layers.Conv1D(numFilters, kernel_size=3, padding='valid',
                activation='relu', dilation_rate=2**(i+1), name='bias_conv_{0:d}'.format(i))(prevLayer)
        prevLength = int_shape(prevLayer)[1]
        newLength = int_shape(newConv)[1]
        newCrop = keras.layers.Cropping1D((prevLength - newLength) //2, 
                name='bias_crop{0:d}'.format(i))(prevLayer)
        prevLayer = keras.layers.add([newConv, newCrop], name='bias_add{0:d}'.format(i))
    countsOutputs = []
    profileOutputs = []
    for individualHead in headList:
        h = _biasModelHead(prevLayer, individualHead, outputFilterWidth)
        countsOutputs.append(h[1])
        profileOutputs.append(h[0])
    m = keras.Model(inputs=inputLayer, outputs = profileOutputs + countsOutputs, name="bias_model")
    return m



def _buildSimpleRegressionModel(architectureSpecification, inputShape, headName, inputLayer):
    activationLayers = []
    for layerType in architectureSpecification["types"]:
        match layerType:
            case "linear":
                activationLayers.append(layers.LinearRegression(name='regress_linear_{0:s}'.format(headName))(inputLayer))
            case 'sigmoid':
                inputLinear = layers.LinearRegression(name='sigmoid_in_linear_{0:s}'.format(headName))(inputLayer)
                sigmoided = keras.layers.Activation(activation=keras.activations.sigmoid, name='sigmoid_activation_{0:s}'.format(headName))(inputLinear)
                outputLinear = layers.LinearRegression(name='sigmoid_out_linear_{0:s}'.format(headName))(sigmoided)
                activationLayers.append(outputLinear)
            case _:
                raise ValueError("The simple layer type you gave ({0:s}) is not supported".format(layerType))
    if(len(activationLayers) > 1):
        sumLayer = keras.layers.Add(name='regress_sum_{0:s}'.format(headName))(activationLayers)
    else:
        sumLayer = activationLayers[0]
    return sumLayer


def _regressionHead(biasProfile, biasCounts, individualHead, architectureSpecification):
    numOutputs = len(individualHead["data"])
    match architectureSpecification["name"]:
        case 'simple':
            profileRegression = _buildSimpleRegressionModel(architectureSpecification, 
                    (architectureSpecification['output-length'], numOutputs), 
                    individualHead["head-name"] + "_profile",
                    biasProfile)
            countsRegression = _buildSimpleRegressionModel(architectureSpecification, 
                    (numOutputs,),
                    individualHead["head-name"] + "_counts",
                    biasCounts)
        case _:
            raise ValueError("Currently, only simple regression is supported.")


    return (profileRegression, countsRegression)



def regressionModel(biasModel, architectureSpecification, headList):
    """Given a Keras model for the bias, generate a simple network that can be used to transform the bias into the experimental data.
    That is, 
    experimental = f(bias)
    and f is a simple function like y=mx+b or something.
    When you train the model returned by this function, you are training the m and b parameters of that function.
    Note that this function sets the bias model to non-trainable, since you're not trying to make the bias model better, you're
    trying to transform the bias to look like experimental data."""

    biasModel.trainable=False
    profileOutputs = []
    countsOutputs = []
    numHeads = len(headList)
    for i, individualHead in enumerate(headList):
        profileHead, countsHead = _regressionHead(
                biasModel.outputs[i], 
                biasModel.outputs[i+numHeads],
                individualHead,
                architectureSpecification)
        profileOutputs.append(profileHead)
        countsOutputs.append(countsHead)
    m = keras.Model(inputs=biasModel.input, outputs = profileOutputs + countsOutputs, name="regression_model")
    return m

def combinedModel(inputLength, outputLength, numFilters, numLayers, inputFilterWidth, outputFilterWidth, headList, regressionModel):
    """This builds a standard bpnet model, but then adds in the bias at the very end: 
            ,-----------------SEQUENCE------------------,
            V                                           V
        _____________                           _________________
        | BIAS MODEL|                           | RESIDUAL MODEL|
        |___________|                           |_______________|
             |                                          |
        _____V_______          _______                  |
        |REGRESSION |--------> | ADD |<-----------------'
        |___________|          |_____|
                                  |
                             _____V_____
                             |COMBINED |
                             |_________|
    Since you'll usually want to isolate the bias-free model, that is returned separately. 
    Parameters:
    inputLength is the length of the one-hot encoded DNA sequence (which must be the same for the bias model and the residual model). 
    outputLength is the width of the predicted profile. 
    numFilters is the number of convolutional filters used at each layer in the residual model. 
    numLayers is the number of dilated convolutions in the residual model. 
    inputFilterWidth is the width of the first convolutional layer in the residual model, the one looking for motifs. 
    outputFilterWidth is the width of the profile head convolutional filter in the residual model at the very bottom of the network.
    taskInfo is taken directly from a <bigwig-list> in the configuration JSON.

    Returns two Keras models. The first is the combined output, i.e., the COMBINED node in the graph above.  
    Input to this model is a (batch x inputLength x 4) tensor of one-hot encoded DNA.
    Output is a list of (profilePreds, countPreds)
    profilePreds is a tensor of shape (batch x numTasks x outputLength), containing the 
    logits of the profile values for each task. 
    countsPreds is a tensor of shape (batch x numTasks) containing the log counts for each task. 
    The second is the bias-free model, RESIDUAL MODEL in the graph above. It has the same input and output shapes as the COMBINED model. 


    It is an error to call this function with an inconsistent network structure, such as an input that is too long.
    """
    regressionModel.trainable=False
    inputLayer = keras.Input((inputLength, 4), name='residual_input')

    initialConv = keras.layers.Conv1D(numFilters, kernel_size=inputFilterWidth, padding='valid',
            activation='relu', name="residual_initial_conv")(inputLayer)
    prevLayer = initialConv
    for i in range(numLayers):
        newConv = keras.layers.Conv1D(numFilters, kernel_size=3, padding='valid',
                activation='relu', dilation_rate=2**(i+1), name='residual_conv_{0:d}'.format(i))(prevLayer)
        prevLength = int_shape(prevLayer)[1]
        newLength = int_shape(newConv)[1]
        newCrop = keras.layers.Cropping1D((prevLength - newLength) //2, 
                name='residual_crop{0:d}'.format(i))(prevLayer)
        prevLayer = keras.layers.add([newConv, newCrop], name='residual_add{0:d}'.format(i))
    countsOutputs = []
    profileOutputs = []
    for individualHead in headList:
        h = _biasModelHead(prevLayer, individualHead, outputFilterWidth)
        countsOutputs.append(h[1])
        profileOutputs.append(h[0])
    bpnetModel = keras.Model(inputs=inputLayer, outputs = profileOutputs + countsOutputs, name="residual_model")
    readyRegressionModel = regressionModel(inputLayer)
    
    combinedProfileHeads = []
    combinedCountsHeads = []
    numHeads = len(headList)
    for i, individualHead in enumerate(headList):
        addProfile = keras.layers.Add(name='combined_add_profile_{0:s}'.formatindividualHead["head-name"])([readyRegressionModel.outputs[i], bpnetModel.outputs[i]])
        addCounts = keras.layers.Add(name='combined_add_counts_{0:s}'.formatindividualHead["head-name"])([readyRegressionModel.outputs[i+numHeads], bpnetModel.outputs[i + numHeads]])
        combinedProfileHeads.append(addProfile)
        combinedCountsHeads.append(addCounts)
    combinedModel = keras.Model(inputs=inputLayer, outputs = combinedProfileHeads + combinedCountsHeads, name='combined_model')
    return (combinedModel, bpnetModel)


