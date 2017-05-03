ifeq ($(shell hostname),hyades.ucsc.edu)                                                                      
	CC=icpc -std=c++11 -I/pfs/sw/serial/gcc/boost-1.57.0/include -I$(PYTHIA8)/include
	#LFLAGS=/pfs/sw/serial/gcc/boost-1.57.0/lib/libboost_program_options.so.1.57.0
	LFLAGS=-L$(PYTHIA8)/lib -lpythia8 -L/pfs/sw/serial/gcc/boost-1.57.0/lib/ -lboost_thread -lboost_system
else
	CC=clang++ -std=c++11
	#LFLAGS=-lboost_program_options
	LFLAGS=-lpythia8 -lboost_thread -lboost_system
endif

SRCDIR=src
BINDIR=bin

GenAntimatter:
	$(CC) $(LFLAGS) $(SRCDIR)/$@.cpp -o $(BINDIR)/$@

clean:
	rm $(BINDIR)/*
