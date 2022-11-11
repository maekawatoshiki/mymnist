SRCS=$(wildcard *.c)
OBJS=$(SRCS:%.c=%.o)
DEPS=$(SRCS:%.c=%.d)
CFLAGS=-O3 -march=native # -mtune=native -ftree-vectorize -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5
LDFLAGS=-lblis
CC=gcc

all: mnist

mnist: $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.c
	# clang-format-11 -i $<
	$(CC) -MMD -MP $(CFLAGS) -c -o $@ $<

test: all
	./mnist

clean:
	rm -rf mnist $(OBJS) $(DEPS)

-include $(DEPS)

.PHONY: all test clean
