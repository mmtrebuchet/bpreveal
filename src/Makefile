
jaccard.cpython-311-x86_64-linux-gnu.so: libjaccard.c libjaccard.pyf
	CFLAGS="-Ofast -march=native -fexpensive-optimizations -ffast-math" f2py -c libjaccard.pyf libjaccard.c

clean: jaccard.cpython-311-x86_64-linux-gnu.so
	rm $<

all: jaccard.cpython-311-x86_64-linux-gnu.so
