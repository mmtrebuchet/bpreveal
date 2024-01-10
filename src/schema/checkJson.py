#!/usr/bin/env python3
import jsonschema
import json
import argparse

parser = argparse.ArgumentParser(description="Check to see if a json file has schema problems.")
parser.add_argument("-s", "--schema-name",
    help="The name of the schema, like prepareBed. If omitted, check all schemas.",
    dest="schemaName")

parser.add_argument("jsons", help="The name of the json files to validate.", nargs='+')

args = parser.parse_args()


from bpreveal.schema import prepareBed, prepareTrainingData, trainSoloModel, \
    trainTransformationModel, trainCombinedModel, makePredictionsBed, interpretFlat, \
    makePredictionsFasta, interpretPisa

schemaMap = {"prepareBed": prepareBed,
             "prepareTrainingData": prepareTrainingData,
             "trainSoloModel": trainSoloModel,
             "trainTransformationModel": trainTransformationModel,
             "trainCombinedModel": trainCombinedModel,
             "makePredictionsBed": makePredictionsBed,
             "interpretFlat": interpretFlat,
             "makePredictionsFasta": makePredictionsFasta,
             "interpretPisa": interpretPisa}

for jsonFname in args.jsons:
    with open(jsonFname, "r") as fp:
        testJson = json.load(fp)
    if args.schemaName is not None:
        jsonschema.validate(schema=schemaMap[args.schemaName],
                            instance=testJson)
        print(jsonFname, "Validated.")
    else:
        print(jsonFname)
        for schema in schemaMap.keys():
            try:
                jsonschema.validate(schema=schemaMap[schema], instance=testJson)
                print("    Validated as ", schema)
            except:
                pass
