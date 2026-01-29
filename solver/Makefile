# Compiler and flags
CC = /opt/homebrew/bin/gcc-15
CFLAGS = -std=c11 -O3 -march=native -fopenmp
INCLUDES = -I/opt/homebrew/opt/openblas/include -I/opt/homebrew/opt/suite-sparse/include -I/opt/homebrew/opt/suite-sparse/include/suitesparse -Iinclude
LDFLAGS = -L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/opt/suite-sparse/lib -lm -lopenblas -lcholmod

# Source and object files
SRC_DIR = src
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,%.o,$(filter-out $(SRC_DIR)/top3d.c,$(SRCS)))

# Output binary
TARGET = top3d

# Build rules
all: $(TARGET)

%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

$(TARGET): $(SRC_DIR)/top3d.c $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
