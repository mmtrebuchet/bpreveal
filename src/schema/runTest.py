#!/usr/bin/env python3
import jsonschema
import json


def runTest(schema, jsonFname, good):
    success = True
    with open(jsonFname, "r") as fp:
        testJson = json.load(fp)
    try:
        jsonschema.validate(instance=testJson, schema=schema)
    except:
        success = False
        if good:
            raise
    if not good and success:
        raise Exception(jsonFname + " validated but shouldn't have.")
    print(jsonFname + " tested successfully.")


from bpreveal.schema import prepareBed, prepareTrainingData, trainSoloModel,\
    trainTransformationModel, trainCombinedModel, makePredictionsBed, interpretFlat
for goodIdx in range(2):
    runTest(prepareBed, "testcases/prepareBed_good_{0:d}.json".format(goodIdx), True)
for goodIdx in range(3):
    runTest(prepareBed, "testcases/prepareBed_bad_{0:d}.json".format(goodIdx), False)

for goodIdx in range(2):
    runTest(prepareTrainingData,
            "testcases/prepareTrainingData_good_{0:d}.json".format(goodIdx), True)


for goodIdx in range(1):
    runTest(trainSoloModel, "testcases/trainSoloModel_good_{0:d}.json".format(goodIdx), True)
for badIdx in range(1):
    runTest(trainSoloModel, "testcases/trainSoloModel_bad_{0:d}.json".format(badIdx), False)


for goodIdx in range(1):
    runTest(trainTransformationModel,
            "testcases/trainTransformationModel_good_{0:d}.json".format(goodIdx), True)

for goodIdx in range(1):
    runTest(trainCombinedModel,
            "testcases/trainCombinedModel_good_{0:d}.json".format(goodIdx), True)
for badIdx in range(1):
    runTest(trainCombinedModel,
            "testcases/trainCombinedModel_bad_{0:d}.json".format(badIdx), False)

for goodIdx in range(1):
    runTest(makePredictionsBed,
            "testcases/makePredictionsBed_good_{0:d}.json".format(goodIdx), True)

for goodIdx in range(1):
    runTest(interpretFlat,
            "testcases/interpretFlat_good_{0:d}.json".format(goodIdx), True)
