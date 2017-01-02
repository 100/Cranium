LIBS = -lm
FLAGS = -Wall -o
COMPILER = gcc

tests: matrix_tests function_tests layer_tests network_tests optimizer_tests

matrix_tests:
	$(COMPILER) $(FLAGS) matrix_tests matrix_tests.c
	./matrix_tests
	rm matrix_tests

function_tests:
	$(COMPILER) $(FLAGS) function_tests function_tests.c
	./function_tests
	rm function_tests

layer_tests:
	$(COMPILER) $(FLAGS) layer_tests layer_tests.c
	./layer_tests
	rm layer_tests

network_tests:
	$(COMPILER) $(FLAGS) network_tests network_tests.c
	./network_tests
	rm network_tests

optimizer_tests:
	$(COMPILER) $(FLAGS) optimizer_tests optimizer_tests.c
	./optimizer_tests
	rm optimizer_tests
