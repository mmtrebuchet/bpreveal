#!/usr/bin/env python3
import sys
with open(sys.argv[1], "w") as fp:
    fp.write("import json\n\n")
    for schemaFname in sys.argv[2:]:
        fp.write(schemaFname + ' = json.loads("""')
        with open("schema/" + schemaFname + ".schema", "r") as sfp:
            for line in sfp:
                fp.write(line)
        fp.write('""")\n\n\n')
