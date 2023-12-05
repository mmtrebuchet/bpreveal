.DEFAULT_GOAL := all
internal/libushuffle.cpython-311-x86_64-linux-gnu.so: libushuffle.cpython-311-x86_64-linux-gnu.so
	mv $^ $@
libushuffle.cpython-311-x86_64-linux-gnu.so: internal/libushuffle.c internal/libushuffle.pyf
	CFLAGS="-Ofast -march=native -fexpensive-optimizations -ffast-math" f2py -c $^

jaccard.cpython-311-x86_64-linux-gnu.so: internal/libjaccard.c internal/libjaccard.pyf
	CFLAGS="-Ofast -march=native -fexpensive-optimizations -ffast-math" f2py -c $^

clean: jaccard.cpython-311-x86_64-linux-gnu.so internal/libushuffle.cpython-311-x86_64-linux-gnu.so
	rm $^

all: jaccard.cpython-311-x86_64-linux-gnu.so internal/libushuffle.cpython-311-x86_64-linux-gnu.so
