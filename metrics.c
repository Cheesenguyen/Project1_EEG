#include "metrics.h"
#include "utils.h"
#include <string.h>

void metrics_reset(Metrics *m){ memset(m, 0, sizeof(*m)); }

static void fmt_bytes(char *buf, long bytes){
    if      (bytes >= 1024*1024) sprintf(buf, "%.2f MB", bytes / (1024.0*1024.0));
    else if (bytes >= 1024)      sprintf(buf, "%.2f KB", bytes / 1024.0);
    else                         sprintf(buf, "%ld B",   bytes);
}

void metrics_print_terminal(int n, const Config *cfgs, const Metrics *ms)
{
    printf("CONVOLUTION ARCHITECTURE SIMULATOR — RESULTS\n");

    for (int i = 0; i < n; i++) {
        const Config  *c = &cfgs[i];
        const Metrics *m = &ms[i];

        char si[20], sw[20], st[20];
        fmt_bytes(si, m->sram_input_bytes);
        fmt_bytes(sw, m->sram_weight_bytes);
        fmt_bytes(st, m->sram_total_bytes);

        printf("\n");
        printf("               Config\n");
        printf("  %-24s : %d x %d x %d\n",     "Input  (P x Q x C)",   c->P, c->Q, c->C);
        printf("  %-24s : %d x %d x %d\n",     "Output (H x W x K)",   c->H, c->W, c->K);
        printf("  %-24s : %d x %d\n",           "Kernel (R x S)",       c->R, c->S);
        printf("  %-24s : %d\n",                "Stride",               c->stride);
        printf("  %-24s : %d\n",                "Padding",              c->padding);
        printf("  %-24s : %s\n",                "Strategy",             strat_str(c->strategy));
        printf("  %-24s : %d x %d x %d x %d\n","Tile (TH,TW,TC,TK)",   c->TH, c->TW, c->TC, c->TK);
        printf("\n");

        printf("          Table A : Memory Hierarchy\n");
        printf("  %-24s : %ld\n",  "DRAM Loads",         m->dram_loads);
        printf("  %-24s : %ld\n",  "  Input Loads",      m->dram_input_loads);
        printf("  %-24s : %ld\n",  "  Weight Loads",     m->dram_weight_loads);
        printf("  %-24s : %ld\n",  "DRAM Stores",        m->dram_stores);
        printf("  %-24s : %s\n",   "SRAM Input",         si);
        printf("  %-24s : %s\n",   "SRAM Weight",        sw);
        printf("  %-24s : %s\n",   "SRAM Total",         st);
        printf("\n");

        printf("         Table B : Compute Architecture\n");
        printf("  %-24s : %ld\n",  "Max PE",             m->max_PE);
        printf("  %-24s : %ld\n",  "Num PS Buffers",     m->num_ps_buffers);
        printf("  %-24s : %ld\n",  "SRAM Input Reads",   m->sram_input_reads);
        printf("  %-24s : %ld\n",  "SRAM Weight Reads",  m->sram_weight_reads);
        printf("  %-24s : %ld\n",  "Total Cycles",       m->total_cycles);
        printf("\n");
    }
}

void metrics_write_csv(FILE *fp, int n, const Config *cfgs, const Metrics *ms,
                       const char *unroll_mode)
{
    fprintf(fp,
        "label,strategy,unroll,"
        "TH,TW,TC,TK,"
        "dram_loads,dram_input_loads,dram_weight_loads,dram_stores,"
        "sram_input_bytes,sram_weight_bytes,sram_total_bytes,"
        "max_PE,num_ps_buffers,"
        "sram_input_reads,sram_weight_reads,total_cycles\n");

    for (int i = 0; i < n; i++) {
        const Config  *c = &cfgs[i];
        const Metrics *m = &ms[i];

        fprintf(fp,
           "\"%s\",%s,%s,"
            "%d,%d,%d,%d,"
            "%ld,%ld,%ld,%ld,"
            "%ld,%ld,%ld,"
            "%ld,%ld,"
            "%ld,%ld,%ld\n",
            c->label, strat_str(c->strategy), unroll_mode,
            c->TH, c->TW, c->TC, c->TK,
            m->dram_loads, m->dram_input_loads, m->dram_weight_loads, m->dram_stores,
            m->sram_input_bytes, m->sram_weight_bytes, m->sram_total_bytes,
            m->max_PE, m->num_ps_buffers,
            m->sram_input_reads, m->sram_weight_reads, m->total_cycles);
    }
}