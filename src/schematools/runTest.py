#!/usr/bin/env python3
"""Make sure that all of the schemas are correct by testing them on known good and bad inputs."""
# flake8: noqa: T201
import json
import os
import argparse
import jsonschema
import bpreveal.schema as schemas
from bpreveal.internal import interpreter
p = argparse.ArgumentParser(description="Check the test cases for schemas.")
p.add_argument("--show-correct", help="Show a note when a test is successful.",
               action="store_true", dest="showCorrect")
args = p.parse_args()


def runTest(schema, jsonFname, good):
    """Given a schema and a json name, check whether it passes or not.

    Runs the test against the given schema, and also checks all other
    BPReveal schemas to make sure the json doesn't validate as any of them.

    If good is True, then it is an error if it fails to pass.
    If good is False, then it is an error if it passes.
    """
    dats = interpreter.evalFile("testcases/" + jsonFname)
    try:
        schemas.schemaMap[schema].validate(dats)
        if good:
            if args.showCorrect:
                print(f"    \u2713 Good, pass {jsonFname}, {schema}")
        else:
            print(f"\u2612 Bad, pass {jsonFname}, {schema}")
    except jsonschema.ValidationError:
        if good:
            print(f"\u2717 Good, fail {jsonFname}, {schema}")
        elif args.showCorrect:
            print(f"    \u2611 Bad, fail {jsonFname}, {schema}")
    for otherSchemaName, otherSchema in schemas.schemaMap.items():
        if otherSchemaName == schema:
            continue
        try:
            otherSchema.validate(dats)
            print(f"\u29B8 {jsonFname}, {otherSchemaName}")
        except jsonschema.ValidationError:
            pass


if args.showCorrect:
    print("\u2713 = good json, passed schema.")
    print("\u2611 = bad json, failed schema.")
print("\u2612 = bad json, passed schema.")
print("\u2717 = good json, failed schema.")
print("\u29B8 = passed wrong schema.")

for f in os.listdir("testcases"):
    s, goodBad, _ = f.split("_")
    runTest(s, f, goodBad == "good")
# Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
