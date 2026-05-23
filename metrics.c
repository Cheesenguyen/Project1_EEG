#include "metrics.h"
#include "utils.h"
#include <string.h>

void metrics_reset(Metrics *m){ memset(m, 0, sizeof(*m)); }

/* ── human-readable byte sizes ── */
static void fmt_bytes(char *buf, long bytes){
    if      (bytes >= 1024*1024) sprintf(buf, "%.2f MB", bytes / (1024.0*1024.0));
    else if (bytes >= 1024)      sprintf(buf, "%.2f KB", bytes / 1024.0);
    else                         sprintf(buf, "%ld B",   bytes);
}

 // TERMINAL OUTPUT — liệt kê tham số từng case
void metrics_print_terminal(int n, const Config *cfgs, const Metrics *ms)
{
    printf("CONVOLUTION ARCHITECTURE SIMULATOR — RESULTS\n");

    for (int i = 0; i < n; i++) {
        const Config  *c = &cfgs[i];
        const Metrics *m = &ms[i];

        char si[20], sw[20], sp[20];
        fmt_bytes(si, m->sram_input_size);
        fmt_bytes(sw, m->sram_weight_size);
        fmt_bytes(sp, m->sram_psum_size);

        int OH = out_dim(c->H, c->R, c->padding, c->stride);
        int OW = out_dim(c->W, c->S, c->padding, c->stride);

        printf("\n");

        printf("               Config\n");
        printf("  %-22s : %d x %d x %d\n",   "Input (H x W x C)",   c->H, c->W, c->C);
        printf("  %-22s : %d x %d x %d\n",   "Output (OH x OW x K)", OH,  OW,  c->K);
        printf("  %-22s : %d x %d\n",         "Kernel (R x S)",       c->R, c->S);
        printf("  %-22s : %d\n",              "Stride",               c->stride);
        printf("  %-22s : %d\n",              "Padding",              c->padding);
        printf("  %-22s : %s\n",              "Strategy",             strat_str(c->strategy));
        printf("  %-22s : %d x %d x %d x %d\n","Tile (TH,TW,TC,TK)", c->TH, c->TW, c->TC, c->TK);
        printf("\n");

        printf("          Table A : Memory Hierarchy \n");
        printf("  %-22s : %ld\n",  "DRAM Loads",    m->dram_loads);
        printf("  %-22s : %ld\n",  "DRAM Stores",   m->dram_stores);
        printf("  %-22s : %s\n",   "SRAM Input",    si);
        printf("  %-22s : %s\n",   "SRAM Weight",   sw);
        printf("  %-22s : %s\n",   "SRAM Psum",     sp);
        printf("\n");

        printf("         Table B : Compute Architecture \n");
        printf("  %-22s : %ld\n",  "Max PE",              m->max_PE);
        printf("  %-22s : %ld\n",  "SRAM Input Reads",    m->sram_input_reads);
        printf("  %-22s : %ld\n",  "SRAM Weight Reads",   m->sram_weight_reads);
        printf("  %-22s : %ld\n",  "PS Buffer Reads",     m->ps_buffer_reads);
        printf("  %-22s : %ld\n",  "PS Buffer Writes",    m->ps_buffer_writes);
        printf("  %-22s : %ld\n",  "Total Cycles",        m->total_cycles);
        printf("\n");
    }
}


 // CSV OUTPUT — một dòng header + một dòng mỗi case

void metrics_write_csv(FILE *fp, int n, const Config *cfgs, const Metrics *ms,
                       const char *unroll_mode)
{
    /* Header */
    fprintf(fp,
        "label,strategy,unroll,"
        "H,W,C,K,R,S,stride,padding,"
        "TH,TW,TC,TK,"
        "OH,OW,"
        "dram_loads,dram_stores,"
        "sram_input_bytes,sram_weight_bytes,sram_psum_bytes,"
        "max_PE,"
        "sram_input_reads,sram_weight_reads,"
        "ps_buffer_reads,ps_buffer_writes,"
        "total_cycles\n");

    for (int i = 0; i < n; i++) {
        const Config  *c = &cfgs[i];
        const Metrics *m = &ms[i];

        int OH = out_dim(c->H, c->R, c->padding, c->stride);
        int OW = out_dim(c->W, c->S, c->padding, c->stride);

        fprintf(fp,
            "\"%s\",%s,%s,"
            "%d,%d,%d,%d,%d,%d,%d,%d,"
            "%d,%d,%d,%d,"
            "%d,%d,"
            "%ld,%ld,"
            "%ld,%ld,%ld,"
            "%ld,"
            "%ld,%ld,"
            "%ld,%ld,"
            "%ld\n",
            c->label, strat_str(c->strategy), unroll_mode,
            c->H, c->W, c->C, c->K, c->R, c->S, c->stride, c->padding,
            c->TH, c->TW, c->TC, c->TK,
            OH, OW,
            m->dram_loads, m->dram_stores,
            m->sram_input_size, m->sram_weight_size, m->sram_psum_size,
            m->max_PE,
            m->sram_input_reads, m->sram_weight_reads,
            m->ps_buffer_reads, m->ps_buffer_writes,
            m->total_cycles);
    }
}