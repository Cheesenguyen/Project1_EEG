#ifndef CONFIG_H
#define CONFIG_H

/* ── Stationary strategy ── */
typedef enum { WS = 0, IS = 1, OS = 2 } Strategy;

/* ── One simulation case ── */
typedef struct {
    int H, W, C;          /* Input Feature Map dimensions          */
    int K, R, S;          /* Filter: K output-ch, R×S spatial      */
    int TH, TW, TC, TK;   /* Tile sizes                            */
    int stride, padding;
    Strategy strategy;
    char label[64];        /* e.g. "Case 1"                         */
} Config;

// counters
typedef struct {
    /* Table A – Memory Hierarchy */
    long dram_loads;          // tile-loads  from DRAM (input + weight) 
    long dram_stores;         // tile-stores to   DRAM (output)         
    long dram_input_loads;    // input loads riêng
    long dram_weight_loads;   // weight loads riêng
    long sram_input_size;     // bytes for one Input  tile in SRAM         
    long sram_weight_size;    // bytes for one Weight tile in SRAM         
    long sram_psum_size;      // bytes for Psum/Output buffer in SRAM      
    long sram_total_size;     // = input + weight + psum

    /* Table B – Compute / Dataflow */
    long sram_input_reads;  /* times PE array reads from Input  buffer */
    long sram_weight_reads; /* times PE array reads from Weight buffer */
    long ps_buffer_reads;   /* Psum read-backs  from PS buffer         */
    long ps_buffer_writes;  /* Psum spills into PS buffer              */
    long max_PE;            /* peak PE count in one cycle                */
    long total_cycles;      /* total parallel-MAC invocations            */
} Metrics;

#endif /* CONFIG_H */
