.DEFAULT_GOAL := all
internal/libushuffle.cpython-311-x86_64-linux-gnu.so: libushuffle.cpython-311-x86_64-linux-gnu.so
	mv $^ $@
libushuffle.cpython-311-x86_64-linux-gnu.so: internal/libushuffle.c internal/libushuffle.pyf
	CFLAGS="-Ofast -march=native -fexpensive-optimizations -ffast-math" f2py -c $^

jaccard.cpython-311-x86_64-linux-gnu.so: internal/libjaccard.c internal/libjaccard.pyf
	CFLAGS="-Ofast -march=native -fexpensive-optimizations -ffast-math" f2py -c $^

clean: jaccard.cpython-311-x86_64-linux-gnu.so internal/libushuffle.cpython-311-x86_64-linux-gnu.so
	rm $^

defined_schemas = prepareBed prepareBed_old prepareTrainingData trainSoloModel
schema_files = $(addprefix schema/,$(addsuffix .schema,${defined_schemas}))
schema/__init__.py: schema/build.py ${schema_files}
	python3 $< $@ ${defined_schemas}

schemas: schema/__init__.py schema/build.py ${schema_files}

all: jaccard.cpython-311-x86_64-linux-gnu.so internal/libushuffle.cpython-311-x86_64-linux-gnu.so schema/__init__.py
