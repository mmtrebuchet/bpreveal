"""Functions to build BPNet-style models.

The model architectures are generally derived from the basepairmodels
repository, which is released under an MIT-style license. You can find a copy
at ``etc/basepairmodels_license.txt``.

The arithmetic for residual models is derived from ChromBPNet, but the code is
not derived from that project.
"""
from bpreveal.internal import disableTensorflowLogging  # pylint: disable=unused-import # noqa
from tensorflow.keras.backend import int_shape  # type: ignore
import keras.layers as klayers  # type: ignore
import keras.models as kmodels  # type: ignore
from keras import activations
from bpreveal import layers as bprlayers
from bpreveal import logUtils
from bpreveal.internal.constants import NUM_BASES


def _soloModelHead(dilateOutput: klayers.Layer, individualHead: dict,
                   outputFilterWidth: int) -> \
        tuple[klayers.Layer, klayers.Layer]:
    """Create a single output head for a solo model.

    :param dilateOutput: The last dilated convolutional layer of the model.
    :param individualHead: Taken straight from the configuration json.
    :param outputFilterWidth: Also taken from the configuration json.

    :return: A tuple of (profile, counts), each one is a keras Layer.

        * The profile layer is a (batch x) number-of-tracks x output-length tensor
          containing the logits for each track.
        * The counts layer is a (batch x 1) scalar-valued layer containing the total
          counts for the current head.
    """
    headName = individualHead["head-name"]
    logUtils.debug(f"Initializing head {headName}")
    numOutputs = individualHead["num-tasks"]
    profile = klayers.Conv1D(
            filters=numOutputs, kernel_size=outputFilterWidth, padding="valid",  # noqa
            name=f"solo_profile_{headName}")\
        (dilateOutput)  # noqa
    countsGap = klayers.GlobalAveragePooling1D(
            name=f"solo_counts_gap_{headName}")\
        (dilateOutput)  # noqa
    counts = klayers.Dense(
            units=1,  # noqa
            name=f"solo_logcounts_{headName}")\
        (countsGap)  # noqa
    return (profile, counts)


def soloModel(inputLength: int, outputLength: int,
              numFilters: int, numLayers: int, inputFilterWidth: int,
              outputFilterWidth: int, headList: list[dict],
              modelName: str) -> kmodels.Model:
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

    Input to this model is a (batch x inputLength x NUM_BASES) tensor of one-hot encoded DNA.
    Output is a list of (profilePreds, profilePreds, profilePreds,... ,
    countPreds, countPreds, countPreds...).
    profilePreds is a tensor of shape (batch x numTasks x outputLength), containing the
    logits of the profile values for each task.
    countsPreds is a tensor of shape (batch x numTasks) containing the log counts for
    each task.

    It is an error to call this function with an inconsistent network structure,
    such as an input that is too long.
    """
    del outputLength
    logUtils.debug("Building solo model")
    inputLayer = klayers.Input((inputLength, NUM_BASES), name=f"{modelName}_input")

    initialConv = klayers.Conv1D(
            filters=numFilters, kernel_size=inputFilterWidth, padding="valid",  # noqa
            activation="relu", name=f"{modelName}_initial_conv")\
        (inputLayer)  # noqa
    prevLayer = initialConv
    for i in range(numLayers):
        newConv = klayers.Conv1D(
                filters=numFilters, kernel_size=3, padding="valid", activation="relu",  # noqa
                dilation_rate=2 ** (i + 1), name=f"{modelName}_conv_{i}")\
            (prevLayer)  # noqa
        prevLength = int_shape(prevLayer)[1]
        newLength = int_shape(newConv)[1]
        newCrop = klayers.Cropping1D(
                cropping=(prevLength - newLength) // 2,  # noqa
                name=f"{modelName}_crop{i}")\
            (prevLayer)  # noqa
        prevLayer = klayers.add(
            inputs=[newConv, newCrop], name=f"{modelName}_add{i}")
    countsOutputs = []
    profileOutputs = []
    for individualHead in headList:
        h = _soloModelHead(dilateOutput=prevLayer, individualHead=individualHead,
                           outputFilterWidth=outputFilterWidth)
        countsOutputs.append(h[1])
        profileOutputs.append(h[0])
    m = kmodels.Model(inputs=inputLayer,
                      outputs=profileOutputs + countsOutputs,
                      name=f"{modelName}_model")
    return m


def _buildSimpleTransformationModel(architectureSpecification: dict,
                                    headName: str, inputLayer: klayers.Layer) -> \
        kmodels.Model:
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
                    bprlayers.linearRegression(
                            name=f"regress_linear_{headName}")  # noqa
                        (inputLayer))  # noqa
            case "sigmoid":
                inputLinear = bprlayers.linearRegression(
                        name=f"sigmoid_in_linear_{headName}")\
                    (inputLayer)  # noqa
                sigmoided = klayers.Activation(
                        activation=activations.sigmoid,  # noqa
                        name=f"sigmoid_activation_{headName}")\
                    (inputLinear)  # noqa
                outputLinear = bprlayers.linearRegression(
                        name=f"sigmoid_out_linear_{headName}")\
                    (sigmoided)  # noqa
                activationLayers.append(outputLinear)
            case "relu":
                inputLinear = bprlayers.linearRegression(
                        name=f"relu_in_linear_{headName}")\
                    (inputLayer)  # noqa
                sigmoided = klayers.Activation(
                        activation=activations.relu,  # noqa
                        name=f"relu_activation_{headName}")\
                    (inputLinear)  # noqa
                outputLinear = bprlayers.linearRegression(
                        name=f"relu_out_linear_{headName}")\
                    (sigmoided)  # noqa
                activationLayers.append(outputLinear)
            case _:
                raise ValueError(f"The simple layer type you gave ({layerType}) is not supported")
    if len(activationLayers) > 1:
        sumLayer = klayers.Add(
                name=f"regress_sum_{headName}")\
            (activationLayers)  # noqa
    else:
        sumLayer = activationLayers[0]
    return sumLayer


def _transformationHead(soloProfile: klayers.Layer, soloCounts: klayers.Layer,
                        individualHead: dict,
                        profileArchitectureSpecification: dict,
                        countsArchitectureSpecification: dict) -> \
        tuple[klayers.Layer, klayers.Layer]:
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
    headName = individualHead["head-name"]
    logUtils.debug(f"Building transformation head {headName}")
    match profileArchitectureSpecification["name"]:
        case "simple":
            profileTransformation = _buildSimpleTransformationModel(
                architectureSpecification=profileArchitectureSpecification,
                headName=f"profile_{headName}",
                inputLayer=soloProfile)
        case "passthrough":
            profileTransformation = soloProfile
        case _:
            raise ValueError("Currently, only simple regression is supported.")

    match countsArchitectureSpecification["name"]:
        case "simple":
            countsTransformation = _buildSimpleTransformationModel(
                architectureSpecification=countsArchitectureSpecification,
                headName=f"logcounts_{headName}",
                inputLayer=soloCounts)
        case "passthrough":
            countsTransformation = soloCounts
        case _:
            raise ValueError("Currently, only simple regression is supported.")

    return (profileTransformation, countsTransformation)


def transformationModel(soloModelIn: kmodels.Model,
                        profileArchitectureSpecification: dict,
                        countsArchitectureSpecification: dict,
                        headList: list[dict]) -> kmodels.Model:
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
                soloProfile=soloModelIn.outputs[i],  # noqa  # type: ignore
                soloCounts=soloModelIn.outputs[i + numHeads],  # type: ignore
                individualHead=individualHead,
                profileArchitectureSpecification=profileArchitectureSpecification,
                countsArchitectureSpecification=countsArchitectureSpecification)
        profileOutputs.append(profileHead)
        countsOutputs.append(countsHead)
    m = kmodels.Model(inputs=soloModelIn.input,
                      outputs=profileOutputs + countsOutputs,
                      name="transformation_model")
    return m


def combinedModel(inputLength: int, outputLength: int, numFilters: int,
                  numLayers: int, inputFilterWidth: int,
                  outputFilterWidth: int, headList: list[dict],
                  biasModel: kmodels.Model) -> \
        tuple[kmodels.Model, kmodels.Model, kmodels.Model]:
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

    :return: Three kmodels.

        * The first is the combined output, i.e., the COMBINED node in the
          graph above. Input to this model is a (batch x inputLength x NUM_BASES) tensor
          of one-hot encoded DNA. Output is a list of (profilePreds,
          profilePreds, profilePreds,... , countPreds, countPreds,
          countPreds...). profilePreds is a tensor of shape (batch x numTasks x
          outputLength), containing the logits of the profile values for each
          task. countsPreds is a tensor of shape (batch x numTasks) containing
          the log counts for each task.

        * The second is the bias-free model, RESIDUAL MODEL in the graph above.
          It has the same input and output shapes as the COMBINED model.
        * The final model is the solo model, just in case you need it.

    It is an error to call this function with an inconsistent network structure,
    such as an input that is too long.
    """
    # pylint: disable=unsubscriptable-object
    logUtils.debug("Building combined model.")
    biasModel.trainable = False
    residualModel = soloModel(inputLength=inputLength, outputLength=outputLength,
                              numFilters=numFilters, numLayers=numLayers,
                              inputFilterWidth=inputFilterWidth,
                              outputFilterWidth=outputFilterWidth,
                              headList=headList, modelName="residual")
    inputLayer = residualModel.inputs

    assert len(inputLayer) == 1, "Input layer of Keras model is >1. Cannot crop."  # type: ignore
    cropDiff = residualModel.inputs[0].shape[1] - biasModel.inputs[0].shape[1]  # type: ignore
    assert cropDiff % 2 == 0, "Cropping returns an odd number: "\
                              "redo your model input sizes to be even numbers."
    cropFlankSize = int(cropDiff / 2)
    logUtils.info(f"Auto cropdown will trim {cropFlankSize} bases from the bias model input.")
    assert cropFlankSize >= 0, "Bias model inputs are larger than residual inputs"
    croppedInputLayer = klayers.Cropping1D(cropFlankSize,
                                           name="auto_crop_input_tensor")\
                                        (inputLayer[0])  # noqa  # type:ignore
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
        headName = head["head-name"]
        addProfile = klayers.Add(
                name=f"combined_add_profile_{headName}")\
            ([readyBiasHeads[i], residualModel.outputs[i]])  # noqa  # type: ignore
        if head["use-bias-counts"]:
            # While we add logits, we have to convert from log space to linear space
            # This is because we want to model
            # counts = biasCounts + residualCounts noqa
            # but the counts in BPNet are log-counts.
            addCounts = bprlayers.CountsLogSumExp(name=f"combined_logcounts_{headName}")\
                (readyBiasHeads[i + numHeads], residualModel.outputs[i + numHeads])  # noqa
        else:
            # The user doesn't want the counts value from the regression used,
            # just the profile part. This is useful when the concept of a
            # "negative peak set" is meaningless, like in MNase.
            # I use an identity layer so that I can rename it so there's not a spooky
            # 'solo' loss component in a combined model.
            addCounts = klayers.Identity(name=f"combined_logcounts_{headName}")\
                (residualModel.outputs[i + numHeads])  # type: ignore  # noqa
        combinedProfileHeads.append(addProfile)
        combinedCountsHeads.append(addCounts)
    combModel = kmodels.Model(inputs=inputLayer,
                              outputs=combinedProfileHeads + combinedCountsHeads,
                              name="combined_model")
    logUtils.debug("Model built")
    return (combModel, residualModel, readyBiasHeads)
    # pylint: enable=unsubscriptable-object
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
