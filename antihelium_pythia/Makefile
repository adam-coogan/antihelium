CC=clang++ -std=c++11
LFLAGS=-lpythia8 -lboost_thread-mt -lboost_system
SRCDIR=src
BINDIR=bin

GenAntimatter:
	$(CC) $(LFLAGS) $(SRCDIR)/$@.cpp -o $(BINDIR)/$@

clean:
	rm $(BINDIR)/*
