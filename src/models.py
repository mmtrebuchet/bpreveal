"""Functions to build BPNet-style models."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import int_shape
import keras.layers
import keras.models
from bpreveal import layers
from bpreveal import logUtils


def _soloModelHead(dilateOutput: keras.layers.Layer, individualHead: keras.layers.Layer,
                   outputFilterWidth: int) -> \
        tuple[keras.layers.Layer, keras.layers.Layer]:
    """A single output head for a solo model.

    :param dilateOutput: The last dilated convolutional layer of the model.
    :param individualHead: Taken straight from the configuration json.
    :param outputFilterWidth: Also taken from the configuration json.

    :return: A tuple of (profile, counts), each one is a keras Layer.

        * The profile layer is a (batch x) number-of-tracks x output-length tensor
          containing the logits for each track.
        * The counts layer is a (batch x 1) scalar-valued layer containing the total
          counts for the current head.
    """
    logUtils.debug("Initializing head {0:s}".format(individualHead["head-name"]))
    numOutputs = individualHead["num-tasks"]
    profile = keras.layers.Conv1D(
            numOutputs, outputFilterWidth, padding="valid",  # noqa
            name="solo_profile_{0:s}".format(individualHead["head-name"]))\
        (dilateOutput)  # noqa
    countsGap = keras.layers.GlobalAveragePooling1D(
            name="solo_counts_gap_{0:s}".format(individualHead["head-name"]))\
        (dilateOutput)  # noqa
    counts = keras.layers.Dense(
            1,  # noqa
            name="solo_logcounts_{0:s}".format(individualHead["head-name"]))\
        (countsGap)  # noqa
    return (profile, counts)


def soloModel(inputLength: int, outputLength: int,  # pylint: disable=unused-argument
              numFilters: int, numLayers: int, inputFilterWidth: int,
              outputFilterWidth: int, headList: list[dict],
              modelName: str) -> keras.models.Model:
    """Generate a model using the classic BPNet architecture.

    :param inputLength: is the length of the one-hot encoded DNA sequence.
    :param outputLength: is the length of the predicted profile.
    :param numFilters: is the number of convolutional filters used at each layer.
    :param numLayers: is the number of dilated convolutions.
    :param inputFilterWidth: is the width of the first convolutional layer,
        the one looking for motifs.
    :param outputFilterWidth: is the width of the profile head convolutional filter at
        the very bottom of the network.
    :param headList: is taken directly from a <head-list> in the configuration JSON.
    :param modelName: The name you want this model to have when saved.

    :return: A TF model.

    Input to this model is a (batch x inputLength x 4) tensor of one-hot encoded DNA.
    Output is a list of (profilePreds, profilePreds, profilePreds,... ,
    countPreds, countPreds, countPreds...).
    profilePreds is a tensor of shape (batch x numTasks x outputLength), containing the
    logits of the profile values for each task.
    countsPreds is a tensor of shape (batch x numTasks) containing the log counts for
    each task.

    It is an error to call this function with an inconsistent network structure,
    such as an input that is too long.
    """
    logUtils.debug("Building solo model")
    inputLayer = keras.Input((inputLength, 4), name=modelName + "_input")

    initialConv = keras.layers.Conv1D(
            numFilters, kernel_size=inputFilterWidth, padding="valid",  # noqa
            activation="relu", name=modelName + "_initial_conv")\
        (inputLayer)  # noqa
    prevLayer = initialConv
    for i in range(numLayers):
        newConv = keras.layers.Conv1D(
                numFilters, kernel_size=3, padding="valid", activation="relu",  # noqa
                dilation_rate=2 ** (i + 1), name=modelName + "_conv_{0:d}".format(i))\
            (prevLayer)  # noqa
        prevLength = int_shape(prevLayer)[1]
        newLength = int_shape(newConv)[1]
        newCrop = keras.layers.Cropping1D(
                (prevLength - newLength) // 2,  # noqa
                name=modelName + "_crop{0:d}".format(i))\
            (prevLayer)  # noqa
        prevLayer = keras.layers.add(
            [newConv, newCrop], name=modelName + "_add{0:d}".format(i))
    countsOutputs = []
    profileOutputs = []
    for individualHead in headList:
        h = _soloModelHead(prevLayer, individualHead, outputFilterWidth)
        countsOutputs.append(h[1])
        profileOutputs.append(h[0])
    m = keras.Model(inputs=inputLayer,
                    outputs=profileOutputs + countsOutputs,
                    name=modelName + "_model")
    return m


def _buildSimpleTransformationModel(architectureSpecification: dict,
                                    headName: str, inputLayer: keras.layers.Layer) -> \
        keras.models.Model:
    """Actually make the transformation model.

    Builds the model that will transform the tracks produced by the solo model
        (inputLayer) into the experimental data.

    :param architectureSpecification: Taken straight from the config json.
    :param headName: Just a string, used for naming this head.
    :param inputLayer: A keras Layer that will be taken from one of the output heads of
        the solo model. Note that this function is used to transform both the counts and
        profile layers, but it is called once to transform the counts and separately
        to transform the profile.
    :return: A keras Layer that transforms the solo predictions.
    """
    logUtils.debug("Building transformation model.")
    activationLayers = []
    for layerType in architectureSpecification["types"]:
        match layerType:
            case "linear":
                activationLayers.append(
                    layers.LinearRegression(
                            name="regress_linear_{0:s}".format(headName))  # noqa
                        (inputLayer))  # noqa
            case "sigmoid":
                inputLinear = layers.LinearRegression(
                        name="sigmoid_in_linear_{0:s}".format(headName))\
                    (inputLayer)  # noqa
                sigmoided = keras.layers.Activation(
                        activation=keras.activations.sigmoid,  # noqa
                        name="sigmoid_activation_{0:s}".format(headName))\
                    (inputLinear)  # noqa
                outputLinear = layers.LinearRegression(
                        name="sigmoid_out_linear_{0:s}".format(headName))\
                    (sigmoided)  # noqa
                activationLayers.append(outputLinear)
            case "relu":
                inputLinear = layers.LinearRegression(
                        name="relu_in_linear_{0:s}".format(headName))\
                    (inputLayer)  # noqa
                sigmoided = keras.layers.Activation(
                        activation=keras.activations.relu,  # noqa
                        name="relu_activation_{0:s}".format(headName))\
                    (inputLinear)  # noqa
                outputLinear = layers.LinearRegression(
                        name="relu_out_linear_{0:s}".format(headName))\
                    (sigmoided)  # noqa
                activationLayers.append(outputLinear)
            case _:
                raise ValueError("The simple layer type you gave ({0:s}) is not supported"
                        .format(layerType))  # noqa
    if len(activationLayers) > 1:
        sumLayer = keras.layers.Add(
                name="regress_sum_{0:s}".format(headName))\
            (activationLayers)  # noqa
    else:
        sumLayer = activationLayers[0]
    return sumLayer


def _transformationHead(soloProfile: keras.layers.Layer, soloCounts: keras.layers.Layer,
                        individualHead: dict,
                        profileArchitectureSpecification: dict,
                        countsArchitectureSpecification: dict) -> \
        tuple[keras.layers.Layer, keras.layers.Layer]:
    """Make a head for the transformation model.

    Takes the predicted profile and counts Layers from the solo model, and generates a
        head for the transformation model from them.

    :param soloProfile: A keras Layer representing the Profile component of a
        particular solo model head.
    :param soloCounts: is a keras Layer representing the Counts prediction of a
        particular solo model head.
    :param individualHead: is taken straight from the configuration json, and
        contains data about weights and the data files to use.
    :param profileArchitectureSpecification: Straight from the config json.
    :param countsArchitectureSpecification: Straight from the config json.
    :return: a tuple of (profile, counts), each one a keras Layer (or similar) that
        can be treated just like the head of a solo model.
    """
    logUtils.debug("Building transformation head {0:s}"
                   .format(individualHead["head-name"]))
    match profileArchitectureSpecification["name"]:
        case "simple":
            profileTransformation = _buildSimpleTransformationModel(
                profileArchitectureSpecification,
                "profile_" + individualHead["head-name"],
                soloProfile)
        case "passthrough":
            profileTransformation = soloProfile
        case _:
            raise ValueError("Currently, only simple regression is supported.")

    match countsArchitectureSpecification["name"]:
        case "simple":
            countsTransformation = _buildSimpleTransformationModel(
                countsArchitectureSpecification,
                "logcounts_" + individualHead["head-name"],
                soloCounts)
        case "passthrough":
            countsTransformation = soloCounts
        case _:
            raise ValueError("Currently, only simple regression is supported.")

    return (profileTransformation, countsTransformation)


def transformationModel(soloModelIn: keras.models.Model,
                        profileArchitectureSpecification: dict,
                        countsArchitectureSpecification: dict,
                        headList: list[dict]) -> keras.models.Model:
    """Construct a simple model used to regress out the solo model from experimental data.

    Given a solo model (typically representing bias), generate a simple network that
    can be used to transform the solo model's output into the experimental data.
    That is,
    experimental = f(bias)
    and f is a simple function like y=mx+b or something.
    When you train the model returned by this function, you are training the m
    and b parameters of that function. Note that this function sets the solo
    model to non-trainable, since you're not trying to make the bias model
    better, you're trying to transform the solo model's output to look like
    experimental data.

    :param soloModelIn: A Keras model that you'd like to transform.
    :param profileArchitectureSpecification: Straight from the config JSON.
    :param countsArchitectureSpecification: Straight from the config JSON.
    :param headList: Also from the config JSON.
    :return: A Keras model with the same output shape as the soloModel.
    """
    soloModelIn.trainable = False
    profileOutputs = []
    countsOutputs = []
    numHeads = len(headList)
    for i, individualHead in enumerate(headList):
        profileHead, countsHead = _transformationHead(
                soloModelIn.outputs[i],  # noqa
                soloModelIn.outputs[i + numHeads],
                individualHead,
                profileArchitectureSpecification,
                countsArchitectureSpecification)
        profileOutputs.append(profileHead)
        countsOutputs.append(countsHead)
    m = keras.Model(inputs=soloModelIn.input,
                    outputs=profileOutputs + countsOutputs,
                    name="transformation_model")
    return m


def combinedModel(inputLength: int, outputLength: int, numFilters: int,
                  numLayers: int, inputFilterWidth: int,
                  outputFilterWidth: int, headList: list[dict],
                  biasModel: keras.models.Model) -> \
        tuple[keras.models.Model, keras.models.Model]:
    """Build a combined model.

    This builds a standard BPNet model, but then adds in the bias at the very end::

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

    Since you'll usually want to isolate the bias-free model (AKA residual model),
    that is returned separately.

    :param inputLength: The length of the one-hot encoded DNA sequence (which must be the
        same for the bias model and the residual model).
    :param outputLength: The length of the predicted profile.
    :param numFilters: The number of convolutional filters used at each layer
        in the residual model.
    :param numLayers: The number of dilated convolutions in the residual model.
    :param inputFilterWidth: The width of the first convolutional layer in the residual
        model, the one looking for motifs.
    :param outputFilterWidth: The width of the profile head convolutional filter in the
        residual model at the very bottom of the network.
    :param headList: Taken straight from the config json.
    :param biasModel: A keras model that goes from sequence to transformed bias.
        This is the file that is saved when you generate the transformation model,
        and internally comprises both the solo model and a transformation.

    :return: Two Keras models.

        * The first is the combined output, i.e., the COMBINED node in the
          graph above. Input to this model is a (batch x inputLength x 4) tensor
          of one-hot encoded DNA. Output is a list of (profilePreds,
          profilePreds, profilePreds,... , countPreds, countPreds,
          countPreds...). profilePreds is a tensor of shape (batch x numTasks x
          outputLength), containing the logits of the profile values for each
          task. countsPreds is a tensor of shape (batch x numTasks) containing
          the log counts for each task.

        * The second is the bias-free model, RESIDUAL MODEL in the graph above.
          It has the same input and output shapes as the COMBINED model.

    It is an error to call this function with an inconsistent network structure,
    such as an input that is too long.
    """
    # pylint: disable=unsubscriptable-object
    logUtils.debug("Building combined model.")
    biasModel.trainable = False
    residualModel = soloModel(inputLength, outputLength, numFilters, numLayers,
                              inputFilterWidth, outputFilterWidth, headList, "residual")
    inputLayer = residualModel.inputs

    assert len(inputLayer) == 1, "Input layer of Keras model is >1. Cannot crop."
    cropDiff = residualModel.inputs[0].shape[1] - biasModel.inputs[0].shape[1]
    assert cropDiff % 2 == 0, "Cropping returns an odd number: "\
                              "redo your model input sizes to be even numbers."
    cropFlankSize = int(cropDiff / 2)
    logUtils.info("Auto cropdown will trim {0:d} bases from the bias model input."
                 .format(cropFlankSize))
    assert cropFlankSize >= 0, "Bias model inputs are larger than residual inputs"
    croppedInputLayer = tf.keras.layers.Cropping1D(cropFlankSize,
                                                   name="auto_crop_input_tensor")\
                                                (inputLayer[0])  # noqa
    readyBiasHeads = biasModel([croppedInputLayer])

    logUtils.debug("Bias heads")
    logUtils.debug(readyBiasHeads)
    logUtils.debug("Residual model")
    logUtils.debug(residualModel)
    # Build up the output heads. Note that the residual model also has the standard
    # array of output heads, this next step is to combine the residual and regression
    # models to generate the combined model.
    combinedProfileHeads = []
    combinedCountsHeads = []
    numHeads = len(headList)
    for i, head in enumerate(headList):
        # Just straight-up add the logit tensors.
        addProfile = keras.layers.Add(
                name="combined_add_profile_{0:s}".format(head["head-name"]))\
            ([readyBiasHeads[i], residualModel.outputs[i]])  # noqa
        if head["use-bias-counts"]:
            # While we add logits, we have to convert from log space to linear space
            # This is because we want to model
            # counts = biasCounts + residualCounts
            # but the counts in BPNet are log-counts.
            # TODO: Rewrite this using tf.math.reduce_logsumexp, since it would avoid
            # some numerical stability problems.
            absBiasCounts = keras.layers.Activation(
                    tf.math.exp,  # noqa
                    name="combined_exponentiate_bias_{0:s}".format(head["head-name"]))\
                (readyBiasHeads[i + numHeads])  # noqa
            absResidualCounts = keras.layers.Activation(
                    tf.math.exp,  # noqa
                    name="combined_exp_residual_{0:s}".format(head["head-name"])) \
                (residualModel.outputs[i + numHeads])  # noqa
            absCombinedCounts = keras.layers.Add(
                    name="combined_add_counts_{0:s}".format(head["head-name"])) \
                ([absBiasCounts, absResidualCounts])  # noqa
            addCounts = keras.layers.Activation(
                    tf.math.log,  # noqa
                    name="combined_logcounts_{0:s}".format(head["head-name"]))\
                (absCombinedCounts)  # noqa
        else:
            # The user doesn't want the counts value from the regression used,
            # just the profile part. This is useful when the concept of a
            # "negative peak set" is meaningless, like in MNase.
            addCounts = residualModel.outputs[i + numHeads]
        combinedProfileHeads.append(addProfile)
        combinedCountsHeads.append(addCounts)
    combModel = keras.Model(inputs=inputLayer,
                            outputs=combinedProfileHeads + combinedCountsHeads,
                            name="combined_model")
    logUtils.debug("Model built")
    return (combModel, residualModel, readyBiasHeads)
    # pylint: enable=unsubscriptable-object
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
