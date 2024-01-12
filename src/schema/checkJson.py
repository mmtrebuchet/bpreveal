#!/usr/bin/env python3
import json
import argparse
import jsonschema
parser = argparse.ArgumentParser(description="Check to see if a json file has schema problems.")
parser.add_argument("-s", "--schema-name",
    help="The name of the schema, like prepareBed. If omitted, check all schemas.",
    dest="schemaName")

parser.add_argument("jsons", help="The name of the json files to validate.", nargs='+')

args = parser.parse_args()


from bpreveal.schema import schemaMap


for jsonFname in args.jsons:
    with open(jsonFname, "r") as fp:
        testJson = json.load(fp)
    if args.schemaName is not None:
        schemaMap[args.schemaName].validate(testJson)
        print(jsonFname, "Validated.")
    else:
        anyPassed = False
        for schema in schemaMap.keys():
            try:
                schemaMap[schema].validate(testJson)
                print("    " + jsonFname + " →", schema)
                anyPassed = True
            except jsonschema.ValidationError:
                pass
        if not anyPassed:
            print(jsonFname + " Failed to validate")
