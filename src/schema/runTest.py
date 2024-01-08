#!/usr/bin/env python3
import jsonschema
import json

def runTest(schema, jsonFname, good):
    success = True
    with open(jsonFname, "r") as fp:
        testJson = json.load(fp)
    try:
        jsonschema.validate(instance = testJson, schema=schema)
    except:
        success = False
        if good:
            raise
    if not good and success:
        raise Exception(jsonFname + " validated but shouldn't have.")
    print(jsonFname + " tested successfully.")
from bpreveal.schema import prepareBed, prepareTrainingData, trainSoloModel
for goodIdx in range(2):
    runTest(prepareBed, "test/prepareBed_good_{0:d}.json".format(goodIdx), True)
for goodIdx in range(3):
    runTest(prepareBed, "test/prepareBed_bad_{0:d}.json".format(goodIdx), False)

for goodIdx in range(2):
    runTest(prepareTrainingData, "test/prepareTrainingData_good_{0:d}.json".format(goodIdx), True)


for goodIdx in range(1):
    runTest(trainSoloModel, "test/trainSoloModel_good_{0:d}.json".format(goodIdx), True)
for badIdx in range(1):
    runTest(trainSoloModel, "test/trainSoloModel_bad_{0:d}.json".format(badIdx), False)
