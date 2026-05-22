#include "metrics.h"
#include "utils.h"
#include <string.h>

void metrics_reset(Metrics *m){ memset(m, 0, sizeof(*m)); }

/* ── human-readable byte sizes ── */
static void fmt_bytes(char *buf, long bytes){
    if      (bytes >= 1024*1024) sprintf(buf,"%.2f MB", bytes/(1024.0*1024.0));
    else if (bytes >= 1024)      sprintf(buf,"%.2f KB", bytes/1024.0);
    else                         sprintf(buf,"%ld B",   bytes);
}

/* ── horizontal line ── */
static void hline(FILE *fp, const int *w, int n){
    fputc('+', fp);
    for(int i=0;i<n;i++){
        for(int j=0;j<w[i]+2;j++) fputc('-',fp);
        fputc('+',fp);
    }
    fputc('\n',fp);
}

/* ════════════════════════════════════════════════════
 * TABLE A  —  Memory Hierarchy
 * ════════════════════════════════════════════════════ */
static const int WA[] = {8, 4, 4, 5, 5, 3, 3, 12, 12, 14, 14, 15};
/*                       Case TH TW TC  TK  R  S  DRAM-L DRAM-S In-SRAM Wt-SRAM Psum  */
#define NA 12

static void print_table_A(FILE *fp, int n, const Config *cfgs, const Metrics *ms){
    fprintf(fp,
        "\n┌─────────────────────────────────────────────────────────────────────────────┐\n"
        "│          TABLE A : MEMORY HIERARCHY  (DRAM <-> SRAM)                       │\n"
        "└─────────────────────────────────────────────────────────────────────────────┘\n\n");

    hline(fp,WA,NA);
    fprintf(fp,"| %-*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
        WA[0],"Case",
        WA[1],"TH", WA[2],"TW", WA[3],"TC", WA[4],"TK",
        WA[5],"R",  WA[6],"S",
        WA[7],"DRAM Loads", WA[8],"DRAM Stores",
        WA[9],"SRAM Input", WA[10],"SRAM Weight", WA[11],"SRAM Psum");
    hline(fp,WA,NA);

    for(int i=0;i<n;i++){
        const Config *c = &cfgs[i];
        const Metrics *m = &ms[i];
        char si[20],sw[20],sp[20];
        fmt_bytes(si, m->sram_input_size);
        fmt_bytes(sw, m->sram_weight_size);
        fmt_bytes(sp, m->sram_psum_size);

        fprintf(fp,"| %-*s | %*d | %*d | %*d | %*d | %*d | %*d | %*ld | %*ld | %*s | %*s | %*s |\n",
            WA[0], c->label,
            WA[1], c->TH,   WA[2], c->TW,
            WA[3], c->TC,   WA[4], c->TK,
            WA[5], c->R,    WA[6], c->S,
            WA[7], m->dram_loads,
            WA[8], m->dram_stores,
            WA[9], si, WA[10], sw, WA[11], sp);
    }
    hline(fp,WA,NA);
}

/* ════════════════════════════════════════════════════
 * TABLE B  —  Compute Architecture
 * ════════════════════════════════════════════════════ */
static const int WB[] = {8, 8, 9, 18, 17, 14, 14, 16};
/*                       Case Strat MaxPE In-Reads Wt-Reads PS-Reads PS-Writes Cycles */
#define NB 8

static void print_table_B(FILE *fp, int n, const Config *cfgs, const Metrics *ms){
    fprintf(fp,
        "\n┌─────────────────────────────────────────────────────────────────────────────┐\n"
        "│          TABLE B : COMPUTE ARCHITECTURE  (PE & DATAFLOW)                   │\n"
        "└─────────────────────────────────────────────────────────────────────────────┘\n\n");

    hline(fp,WB,NB);
    fprintf(fp,"| %-*s | %-*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
        WB[0],"Case",
        WB[1],"Strategy",
        WB[2],"Max PE",
        WB[3],"SRAM Input Reads",
        WB[4],"SRAM Wght Reads",
        WB[5],"PS Buf Reads",
        WB[6],"PS Buf Writes",
        WB[7],"Total Cycles");
    hline(fp,WB,NB);

    for(int i=0;i<n;i++){
        const Config *c = &cfgs[i];
        const Metrics *m = &ms[i];
        fprintf(fp,"| %-*s | %-*s | %*ld | %*ld | %*ld | %*ld | %*ld | %*ld |\n",
            WB[0], c->label,
            WB[1], strat_str(c->strategy),
            WB[2], m->max_PE,
            WB[3], m->sram_input_reads,
            WB[4], m->sram_weight_reads,
            WB[5], m->ps_buffer_reads,
            WB[6], m->ps_buffer_writes,
            WB[7], m->total_cycles);
    }
    hline(fp,WB,NB);
}

/* ════════════════════════════════════════════════════
 * Full report
 * ════════════════════════════════════════════════════ */
void metrics_print_report(FILE *fp, int n, const Config *cfgs, const Metrics *ms){
    fprintf(fp,
        "\n================================================================================\n"
        "              CONVOLUTION ARCHITECTURE SIMULATOR — RESULTS\n"
        "================================================================================\n");

    /* Print config summary */
    fprintf(fp,"\n  Cases simulated: %d\n", n);
    fprintf(fp,"  Fixed input: H=%d  W=%d  C=%d\n\n",
            cfgs[0].H, cfgs[0].W, cfgs[0].C);

    print_table_A(fp, n, cfgs, ms);
    fprintf(fp,"\n");
    print_table_B(fp, n, cfgs, ms);
    fprintf(fp,"\n================================================================================\n\n");
}
