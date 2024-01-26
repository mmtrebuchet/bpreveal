#!/usr/bin/env python3
import jsonschema
import json
import bpreveal.schema
import os

import argparse
p = argparse.ArgumentParser(description="Check the test cases for schemas.")
p.add_argument("--show-correct", help="Show a note when a test is successful.",
               action='store_true', dest='showCorrect')
args = p.parse_args()


def runTest(schema, jsonFname, good):
    with open("testcases/" + jsonFname, "r") as fp:
        dats = json.load(fp)
    try:
        bpreveal.schema.schemaMap[schema].validate(dats)
        if good:
            if args.showCorrect:
                print("    \u2713 Good, pass {0:s}, {1:s}".format(jsonFname, schema))
        else:
            print("\u2612 Bad, pass {0:s}, {1:s}".format(jsonFname, schema))

    except jsonschema.ValidationError:
        if good:
            print("\u2717 Good, fail {0:s}, {1:s}".format(jsonFname, schema))
        else:
            if args.showCorrect:
                print("    \u2611 Bad, fail {0:s}, {1:s}".format(jsonFname, schema))
    for otherSchema in bpreveal.schema.schemaMap.keys():
        if otherSchema == schema:
            continue
        try:
            bpreveal.schema.schemaMap[otherSchema].validate(dats)
            print("\u29B8 {0:s}, {1:s}".format(jsonFname, otherSchema))
        except jsonschema.ValidationError:
            pass


if args.showCorrect:
    print("\u2713 = good json, passed schema.")
    print("\u2611 = bad json, failed schema.")
print("\u2612 = bad json, passed schema.")
print("\u2717 = good json, failed schema.")
print("\u29B8 = passed wrong schema.")

for f in os.listdir("testcases"):
    s, goodBad, _ = f.split('_')
    runTest(s, f, goodBad == "good")
