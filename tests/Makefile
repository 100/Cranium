LIBS = -lm
FLAGS = -std=c99 -Wall -Wno-unused-function -O3 -o
COMPILER = gcc

tests: matrix_tests function_tests layer_tests network_tests optimizer_tests

matrix_tests:
	$(COMPILER) $(FLAGS) matrix_tests matrix_tests.c $(LIBS)
	./matrix_tests
	rm matrix_tests

function_tests:
	$(COMPILER) $(FLAGS) function_tests function_tests.c $(LIBS)
	./function_tests
	rm function_tests

layer_tests:
	$(COMPILER) $(FLAGS) layer_tests layer_tests.c $(LIBS)
	./layer_tests
	rm layer_tests

network_tests:
	$(COMPILER) $(FLAGS) network_tests network_tests.c $(LIBS)
	./network_tests
	rm network_tests
	rm network.pkl
	rm network2.pkl

optimizer_tests:
	$(COMPILER) $(FLAGS) optimizer_tests optimizer_tests.c $(LIBS)
	./optimizer_tests
	rm optimizer_tests
