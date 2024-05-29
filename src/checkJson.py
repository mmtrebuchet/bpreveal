#!/usr/bin/env python3
"""A utility to check your json files for problems.

Useful before you submit a big job!
"""
# flake8: noqa: T201
import json
import argparse
import jsonschema
from bpreveal.schema import schemaMap


def getParser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description="Check to see if a json file has schema problems.")
    parser.add_argument("-s", "--schema-name",
        help="The name of the schema, like prepareBed. If omitted, check all schemas.",
        dest="schemaName")
    parser.add_argument("jsons", help="The name of the json files to validate.", nargs="+")
    return parser


def main():
    """Run the checks."""
    args = getParser().parse_args()
    fnameByMatchedSchema = {}
    failedFnames = []
    for jsonFname in args.jsons:
        with open(jsonFname, "r") as fp:
            testJson = json.load(fp)
        if args.schemaName is not None:
            schemaMap[args.schemaName].validate(testJson)
            print(jsonFname, "Validated.")
        else:
            anyPassed = False
            for schemaName, schema in schemaMap.items():
                try:
                    schema.validate(testJson)
                    # print("    " + jsonFname + " →", schema)
                    anyPassed = True
                    if schemaName not in fnameByMatchedSchema:
                        fnameByMatchedSchema[schemaName] = []
                    fnameByMatchedSchema[schemaName].append(jsonFname)

                except jsonschema.ValidationError:
                    pass
            if not anyPassed:
                print(jsonFname + " Failed to validate")
    for schemaName, matches in fnameByMatchedSchema.items():
        print("    " + schemaName + "")
        for fname in matches:
            print("        →" + fname)
    for fname in failedFnames:
        print(fname + " FAILED to validate")


if __name__ == "__main__":
    main()
# Copyright 2022, 2023, 2024 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.  # noqa
