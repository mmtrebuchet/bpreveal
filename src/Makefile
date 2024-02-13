.DEFAULT_GOAL := all
internal/libushuffle.cpython-311-x86_64-linux-gnu.so: libushuffle.cpython-311-x86_64-linux-gnu.so
	mv $^ $@
internal/libjaccard.cpython-311-x86_64-linux-gnu.so: libjaccard.cpython-311-x86_64-linux-gnu.so
	mv $^ $@

libushuffle.cpython-311-x86_64-linux-gnu.so: internal/libushuffle.c internal/libushuffle.pyf
	CFLAGS="-Ofast -fexpensive-optimizations -ffast-math" f2py -c $^

libjaccard.cpython-311-x86_64-linux-gnu.so: internal/libjaccard.c internal/libjaccard.pyf
	CFLAGS="-Ofast -fexpensive-optimizations -ffast-math" f2py -c $^

clean: internal/libjaccard.cpython-311-x86_64-linux-gnu.so \
	internal/libushuffle.cpython-311-x86_64-linux-gnu.so \
	schema.py
	rm $^

defined_schemas = prepareBed prepareBed_old prepareTrainingData trainSoloModel\
				  trainTransformationModel trainCombinedModel makePredictionsBed\
				  makePredictionsFasta interpretFlat interpretPisa\
				  motifSeqletCutoffs motifScan addNoise base

schema_files = $(addprefix schematools/,$(addsuffix .schema,${defined_schemas}))

schema.py: schematools/build.py ${schema_files}
	python3 $< $@ ${defined_schemas}

schemas: schema.py schematools/build.py ${schema_files}

all: internal/libjaccard.cpython-311-x86_64-linux-gnu.so \
	internal/libushuffle.cpython-311-x86_64-linux-gnu.so \
	schemas
