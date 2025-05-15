.DEFAULT_GOAL := all
internal/libushuffle.cpython-312-x86_64-linux-gnu.so: libushuffle.cpython-312-x86_64-linux-gnu.so
	mv $^ $@
internal/libjaccard.cpython-312-x86_64-linux-gnu.so: libjaccard.cpython-312-x86_64-linux-gnu.so
	mv $^ $@
internal/libslide.cpython-312-x86_64-linux-gnu.so: libslide.cpython-312-x86_64-linux-gnu.so
	mv $^ $@

libushuffle.cpython-312-x86_64-linux-gnu.so: internal/libushuffle.c internal/libushuffle.pyf
	CFLAGS="-Ofast -fexpensive-optimizations -ffast-math" f2py -c $^

libjaccard.cpython-312-x86_64-linux-gnu.so: internal/libjaccard.c internal/libjaccard.pyf
	CFLAGS="-Ofast -fexpensive-optimizations -ffast-math" f2py -c $^

libslide.cpython-312-x86_64-linux-gnu.so: internal/libslide.c internal/libslide.pyf
	CFLAGS="-Ofast -fexpensive-optimizations -ffast-math -fopenmp" f2py -c -lgomp $^

clean: internal/libjaccard.cpython-312-x86_64-linux-gnu.so \
	internal/libushuffle.cpython-312-x86_64-linux-gnu.so \
	internal/libslide.cpython-312-x86_64-linux-gnu.so \
	schema.py
	rm $^

defined_schemas = prepareBed prepareBed_old prepareTrainingData trainSoloModel\
				  trainTransformationModel trainCombinedModel makePredictions\
				  interpretFlat interpretPisa\
				  motifSeqletCutoffs motifScan addNoise\
				  pisaPlot pisaGraph makePisaFigure base

schema_files = $(addprefix schematools/,$(addsuffix .schema,${defined_schemas}))

schema.py: schematools/build.py ${schema_files}
	python3 $< $@ ${defined_schemas}

schemas: schema.py schematools/build.py ${schema_files}

all: internal/libjaccard.cpython-312-x86_64-linux-gnu.so \
	internal/libushuffle.cpython-312-x86_64-linux-gnu.so \
	internal/libslide.cpython-312-x86_64-linux-gnu.so \
	schemas
# Copyright 2022-2025 Charles McAnany. This file is part of BPReveal. BPReveal is free software: You can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. BPReveal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with BPReveal. If not, see <https://www.gnu.org/licenses/>.
