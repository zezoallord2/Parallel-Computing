CXX := mpic++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -O2
TARGET := parallel_mpi
SRC := \
main.cpp \
communication.cpp \
utils.cpp \
algorithms/heat_diffusion.cpp \
algorithms/matrix_mult.cpp
OBJ := $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
