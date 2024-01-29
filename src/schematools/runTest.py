#!/usr/bin/env python3
"""Make sure that all of the schemas are correct by testing them on known good and bad inputs."""
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
    """Given a schema and a json name, check whether it passes or not.

    Runs the test against the given schema, and also checks all other
    BPReveal schemas to make sure the json doesn't validate as any of them.

    If good is True, then it is an error if it fails to pass.
    If good is False, then it is an error if it passes.
    """
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
