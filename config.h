#ifndef CONFIG_H
#define CONFIG_H

// Stationary strategy
typedef enum { WS = 0, IS = 1, OS = 2 } Strategy;

// One simulation case 
typedef struct {
    int P, Q;             // kích thước input. P chiều cao, Q chiều rộng
    int H, W, C;          // kích thước output + input channel   
    int K, R, S;          // K: số filter <=> số kênh đầu ra -> output khối 3D            
    int TH, TW, TC, TK;   // kích thước tile của output, input channel, output channel                              
    int stride, padding;  
    Strategy strategy;
    char label[64];
} Config;

typedef struct {
    // Table A – Thông số sử dụng trong bộ nhỡ
    long dram_loads;          // tổng load từ DRAM = input + weight        
    long dram_input_loads;    // số lần load input tile DRAM -> SRAM        
    long dram_weight_loads;   // số lần load weight tile DRAM -> SRAM       
    long dram_stores;         // số lần store output tile PS -> DRAM         
    long sram_input_bytes;    // tổng bytes input đã kéo vào SRAM           
    long sram_weight_bytes;   // tổng bytes weight đã kéo vào SRAM          
    long sram_total_bytes;    // = sram_input_bytes + sram_weight_bytes      

    // Table B – Thông số liên quan đến việc tính toán
    long max_PE;              // số PE chạy song song mỗi cycle            
    long num_ps_buffers;      // số khối PS đang sử dụng đồng thời          
    long sram_input_reads;    // số lần PE đọc SRAM input (sau reuse)       
    long sram_weight_reads;   // số lần PE đọc SRAM weight (sau reuse)      
    long total_cycles;        // tổng số cycle MAC                          
} Metrics;

#endif 