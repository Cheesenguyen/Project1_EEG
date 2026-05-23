CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c99
TARGET  = conv_sim
SRCS    = main.c parser.c tiling.c spatial.c spatial_thw.c stationary.c metrics.c conv.c
OBJS    = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: all
	./$(TARGET) configs.txt results_tk.csv results_thw.csv conv_stats.csv
	
.PHONY: all clean run