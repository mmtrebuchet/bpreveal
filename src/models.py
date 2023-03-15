import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import int_shape
import layers
import logging


def _soloModelHead(dilateOutput, individualHead, outputFilterWidth):
    """This is a single output head for a solo model.
        Arguments:
        dilateOutput is a keras Layer that is the bottom dilated layer in a BPNet-style architecture.
        individualHead is taken straight from the configuration json.
        outputFilterWidth is also taken from the configuration json.

        Returns:
        A tuple of (profile, counts), each one is a keras Layer.
        The profile layer is a (batch x) number-of-tracks x output-width tensor containing the logits for each track.
        The counts layer is a (batch x 1) scalar-valued layer containing the total counts for the current head.
    """
    logging.debug("Initializing head {0:s}".format(individualHead["head-name"]))
    numOutputs = individualHead["num-tasks"]
    profile = keras.layers.Conv1D(numOutputs, outputFilterWidth, padding='valid', name='solo_profile_{0:s}'.format(individualHead["head-name"]))(dilateOutput)
    countsGap = keras.layers.GlobalAveragePooling1D(name='solo_counts_gap_{0:s}'.format(individualHead["head-name"]))(dilateOutput)
    counts = keras.layers.Dense(1, name='solo_logcounts_{0:s}'.format(individualHead["head-name"]))(countsGap)
    return (profile, counts)



def soloModel(inputLength, outputLength, numFilters, numLayers, inputFilterWidth, outputFilterWidth, headList, modelName):
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
    Output is a list of (profilePreds, profilePreds, profilePreds,... , countPreds, countPreds, countPreds...)
    profilePreds is a tensor of shape (batch x numTasks x outputLength), containing the
    logits of the profile values for each task.
    countsPreds is a tensor of shape (batch x numTasks) containing the log counts for each task.

    It is an error to call this function with an inconsistent network structure, such as an input that is too long.
    """
    logging.debug("Building solo model")
    inputLayer = keras.Input((inputLength, 4), name=modelName + '_input')

    initialConv = keras.layers.Conv1D(numFilters, kernel_size=inputFilterWidth, padding='valid',
            activation='relu', name=modelName +"_initial_conv")(inputLayer)
    prevLayer = initialConv
    for i in range(numLayers):
        newConv = keras.layers.Conv1D(numFilters, kernel_size=3, padding='valid',
                activation='relu', dilation_rate=2**(i+1), name = modelName + '_conv_{0:d}'.format(i))(prevLayer)
        prevLength = int_shape(prevLayer)[1]
        newLength = int_shape(newConv)[1]
        newCrop = keras.layers.Cropping1D((prevLength - newLength) //2,
                name=modelName + '_crop{0:d}'.format(i))(prevLayer)
        prevLayer = keras.layers.add([newConv, newCrop], name=modelName + '_add{0:d}'.format(i))
    countsOutputs = []
    profileOutputs = []
    for individualHead in headList:
        h = _soloModelHead(prevLayer, individualHead, outputFilterWidth)
        countsOutputs.append(h[1])
        profileOutputs.append(h[0])
    m = keras.Model(inputs=inputLayer, outputs = profileOutputs + countsOutputs, name=modelName + "_model")
    return m



def _buildSimpleTransformationModel(architectureSpecification, headName, inputLayer):
    """
        Builds the model that will transform the tracks produced by the solo model (inputLayer) into the experimental data.
        Arguments:
            architectureSpecification is taken straight from the config json.
            headName is just a string, used for naming this head.
            inputLayer is a keras Layer that will be taken from one of the output heads of the solo model. Note that this function is used to transform both the counts and profile layers, but it is called once to transform the counts and separately to transform the profile.
        Returns:
            A keras Layer that transforms the solo predictions.
    """
    logging.debug("Building transformation model.")
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
            case 'relu':
                inputLinear = layers.LinearRegression(name='relu_in_linear_{0:s}'.format(headName))(inputLayer)
                sigmoided = keras.layers.Activation(activation=keras.activations.relu, name='relu_activation_{0:s}'.format(headName))(inputLinear)
                outputLinear = layers.LinearRegression(name='relu_out_linear_{0:s}'.format(headName))(sigmoided)
                activationLayers.append(outputLinear)
            case _:
                raise ValueError("The simple layer type you gave ({0:s}) is not supported".format(layerType))
    if(len(activationLayers) > 1):
        sumLayer = keras.layers.Add(name='regress_sum_{0:s}'.format(headName))(activationLayers)
    else:
        sumLayer = activationLayers[0]
    return sumLayer


def _transformationHead(soloProfile, soloCounts, individualHead, profileArchitectureSpecification, countsArchitectureSpecification):
    """Takes the predicted profile and counts Layers from the solo model, and generates a head for the transformation model from them.
        Arguments:
        soloProfile is a keras Layer representing the Profile component of a particular solo model head.
        soloCounts is a keras Layer representing the Counts prediction of a particular solo model head.
        individualHead is taken straight from the configuration json, and contains data about weights and the data files to use.
        profileArchitectureSpecification and countsArchitectureSpecification are taken straight from the config json.
        returns:
        a tuple of (profile, counts), each one a keras Layer (or similar) that can be treated just like the head of a solo model.

    """
    logging.debug("Building transformation head {0:s}".format(individualHead["head-name"]))
    numOutputs = individualHead["num-tasks"]
    match profileArchitectureSpecification["name"]:
        case 'simple':
            profileTransformation = _buildSimpleTransformationModel(profileArchitectureSpecification,
                    #(profileArchitectureSpecification['output-length'], numOutputs),
                    individualHead["head-name"] + "_profile",
                    soloProfile)
        case 'cropdown':
            toCrop = keras.layers.Cropping1D(cropping=profileArchitectureSpecification["bases-to-trim"],
                    name="cropdown_{0:s}".format(individualHead["head-name"]))
            profileTransformation = toCrop
        case "passthrough":
            profileTransformation = soloProfile
        case _:
            raise ValueError("Currently, only simple regression is supported.")

    match countsArchitectureSpecification["name"]:
        case "simple":
            countsTransformation = _buildSimpleTransformationModel(countsArchitectureSpecification,
                    #(numOutputs,),
                    individualHead["head-name"] + "_counts",
                    soloCounts)
        case "passthrough":
            countsTransformation = soloCounts
        case "cropdown":
            raise ValueError("You cannot crop down the counts head - it's already a scalar!")
        case _:
            raise ValueError("Currently, only simple regression is supported.")

    return (profileTransformation, countsTransformation)



def transformationModel(soloModel, profileArchitectureSpecification, countsArchitectureSpecification, headList):
    """Given a solo model (typically representing bias), generate a simple network that
    can be used to transform the solo model's output into the experimental data.
    That is,
    experimental = f(bias)
    and f is a simple function like y=mx+b or something.
    When you train the model returned by this function, you are training the m and b parameters of that function.
    Note that this function sets the solo model to non-trainable, since you're not trying to make the bias model better, you're
    trying to transform the solo model's output to look like experimental data."""

    soloModel.trainable=False
    profileOutputs = []
    countsOutputs = []
    numHeads = len(headList)
    for i, individualHead in enumerate(headList):
        profileHead, countsHead = _transformationHead(
                soloModel.outputs[i],
                soloModel.outputs[i+numHeads],
                individualHead,
                profileArchitectureSpecification,
                countsArchitectureSpecification)
        profileOutputs.append(profileHead)
        countsOutputs.append(countsHead)
    m = keras.Model(inputs=soloModel.input, outputs = profileOutputs + countsOutputs, name="transformation_model")
    return m




def combinedModel(inputLength, outputLength, numFilters, numLayers, inputFilterWidth, outputFilterWidth, headList, biasModel):
    """This builds a standard bpnet model, but then adds in the bias at the very end:
            ,-----------------SEQUENCE------------------,
            V                                           ,
        Cropdown step                                   V
        _____________                           _________________
        | SOLO MODEL|                           | RESIDUAL MODEL|
        |___________|                           |_______________|
             |                                          |
        _____V_______          _______                  |
        | TRANSFORM |--------> | ADD |<-----------------'
        |___________|          |_____|
                                  |
                             _____V_____
                             |COMBINED |
                             |_________|
    Since you'll usually want to isolate the bias-free model (AKA residual model), that is returned separately.
    Parameters:
    inputLength is the length of the one-hot encoded DNA sequence (which must be the same for the bias model and the residual model).
    outputLength is the width of the predicted profile.
    numFilters is the number of convolutional filters used at each layer in the residual model.
    numLayers is the number of dilated convolutions in the residual model.
    inputFilterWidth is the width of the first convolutional layer in the residual model, the one looking for motifs.
    outputFilterWidth is the width of the profile head convolutional filter in the residual model at the very bottom of the network.
    headList is taken straight from the config json.
    biasModel is a keras model that goes from sequence to transformed bias. This is the file that is saved when you generate the
    transformation model, and internally comprises both the solo model and a transformation.

    Returns two Keras models. The first is the combined output, i.e., the COMBINED node in the graph above.
    Input to this model is a (batch x inputLength x 4) tensor of one-hot encoded DNA.
    Output is a list of (profilePreds, profilePreds, profilePreds,... , countPreds, countPreds, countPreds...)
    profilePreds is a tensor of shape (batch x numTasks x outputLength), containing the
    logits of the profile values for each task.
    countsPreds is a tensor of shape (batch x numTasks) containing the log counts for each task.
    The second is the bias-free model, RESIDUAL MODEL in the graph above. It has the same input and output shapes as the COMBINED model.


    It is an error to call this function with an inconsistent network structure, such as an input that is too long.
    """
    logging.debug("Building combined model.")
    biasModel.trainable=False
    residualModel = soloModel(inputLength, outputLength, numFilters, numLayers, inputFilterWidth, outputFilterWidth, headList, "residual")
    inputLayer = residualModel.inputs

    assert len(inputLayer)==1, 'Input layer of Keras model is >1--this breaks cropdown feature.'
    cropDiff = residualModel.inputs[0].shape[1] - biasModel.inputs[0].shape[1]
    assert cropDiff % 2==0, 'Cropping returns an odd number--redo your model input sizes to be even numbers.'
    cropFlankSize = int((cropDiff)/2)
    assert cropFlankSize>=0, 'Bias model inputs are larger than residual inputs--this is not supported.'
    croppedInputLayer = tf.keras.layers.Cropping1D(cropFlankSize, name = 'auto_crop_input_tensor')(inputLayer[0])
    readyBiasHeads = biasModel([croppedInputLayer])

    logging.debug("Bias heads")
    logging.debug(readyBiasHeads)
    logging.debug("Residual model")
    logging.debug(residualModel)

    #Build up the output heads. Note that the residual model also has the standard array of output heads, this next step is to
    #combine the residual and regression models to generate the combined model.
    combinedProfileHeads = []
    combinedCountsHeads = []
    numHeads = len(headList)
    for i, individualHead in enumerate(headList):
        #Just straight-up add the logit tensors.
        addProfile = keras.layers.Add(name='combined_add_profile_{0:s}'.format(individualHead["head-name"]))\
                ([readyBiasHeads[i], residualModel.outputs[i]])
        if(individualHead["use-bias-counts"]):
            #While we add logits, we have to convert from log space to linear space
            #This is because we want to model
            #counts = biasCounts + residualCounts
            #but the counts in bpnet are log-counts.
            #TODO: Rewrite this using tf.math.reduce_logsumexp, since it would avoid some numerical stability problems.
            absBiasCounts = keras.layers.Activation(tf.math.exp, name='combined_exponentiate_bias_{0:s}'.format(individualHead["head-name"]))\
                    (readyBiasHeads[i+numHeads])
            absResidualCounts = keras.layers.Activation(tf.math.exp, name='combined_exponentiate_residual_{0:s}'.format(individualHead["head-name"])) \
                    (residualModel.outputs[i+numHeads])
            absCombinedCounts = keras.layers.Add(name='combined_add_counts_{0:s}'.format(individualHead["head-name"])) \
                    ([absBiasCounts, absResidualCounts])
            addCounts = keras.layers.Activation(tf.math.log, name='combined_log_counts_{0:s}'.format(individualHead["head-name"]))\
                    (absCombinedCounts)
            #addCounts = keras.layers.Add(name='combined_multiply_counts_{0:s}'.format(individualHead["head-name"]))([readyRegressionHeads[i+numHeads], residualModel.outputs[i + numHeads]])
        else:
            #The user doesn't want the counts value from the regression used, just the profile part.
            #This is useful when the concept of a "negative peak set" is meaningless, like in MNase.
            addCounts = residualModel.outputs[i + numHeads]
        combinedProfileHeads.append(addProfile)
        combinedCountsHeads.append(addCounts)
    combinedModel = keras.Model(inputs=inputLayer, outputs = combinedProfileHeads + combinedCountsHeads, name='combined_model')
    logging.debug("Model built")
    return (combinedModel, residualModel, readyBiasHeads)
