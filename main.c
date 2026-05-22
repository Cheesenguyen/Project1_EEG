#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "parser.h"
#include "tiling.h"
#include "spatial.h"
#include "stationary.h"
#include "metrics.h"
#include "utils.h"
#include "conv.h"

static int validate(const Config *c, int case_num){
    int OH = out_dim(c->H, c->R, c->padding, c->stride);
    int OW = out_dim(c->W, c->S, c->padding, c->stride);
    int ok = 1;
    if(OH <= 0 || OW <= 0){
        fprintf(stderr,"[ERROR] Case %d: output dimension non-positive "
                "(OH=%d OW=%d).\n", case_num, OH, OW);
        ok = 0;
    }
    if(c->TH > OH || c->TW > OW)
        fprintf(stderr,"[WARN]  Case %d: tile larger than output. Clamped.\n", case_num);
    if(c->TC > c->C || c->TK > c->K)
        fprintf(stderr,"[WARN]  Case %d: tile larger than channels. Clamped.\n", case_num);
    return ok;
}

/* fp_conv  : file để ghi conv stats
 * case_idx : 0 = case đầu tiên → ghi header; >0 → chỉ ghi data */
static void simulate(const Config *cfg, Metrics *m, FILE *fp_conv, int case_idx){
    metrics_reset(m);
    run_tiling    (cfg, m);
    run_spatial   (cfg, m);
    run_stationary(cfg, m);

    float *inp = NULL, *wt = NULL, *out = NULL;
    int OH, OW;
    if (conv_forward(cfg, &inp, &wt, &out, &OH, &OW) == 0) {
        conv_print_sample(out, OH, OW, cfg->K, cfg->label);
        if (fp_conv)
            conv_write_stats(fp_conv, out, OH, OW, cfg->K, cfg->label,
                             /*append=*/ case_idx > 0);
        free(inp); free(wt); free(out);
    } else {
        fprintf(stderr, "[WARN] %s: conv_forward OOM -- skipped.\n", cfg->label);
    }
}

int main(int argc, char *argv[])
{
    const char *cfg_file  = (argc > 1) ? argv[1] : "configs.txt";
    const char *csv_file  = (argc > 2) ? argv[2] : "results.csv";
    const char *conv_file = (argc > 3) ? argv[3] : "conv_stats.csv";

    printf("              CONVOLUTION ARCHITECTURE SIMULATOR\n");
    printf("  Config file  : %s\n", cfg_file);
    printf("  Metrics CSV  : %s\n", csv_file);
    printf("  Conv stats   : %s\n\n", conv_file);

    Config cases[MAX_CASES];
    int n = parse_config(cfg_file, cases);
    if(n == 0){
        fprintf(stderr,"[ERROR] No cases to simulate.\n");
        return EXIT_FAILURE;
    }
    printf("  Found %d case(s). Simulating...\n", n);

    /* Mở file conv stats trước vòng lặp */
    FILE *fp_conv = fopen(conv_file, "w");
    if (!fp_conv)
        fprintf(stderr,"[WARN] Cannot open '%s' for conv stats.\n", conv_file);

    Config  valid_cfgs[MAX_CASES];
    Metrics valid_ms  [MAX_CASES];
    Metrics results   [MAX_CASES];
    int valid_count = 0;

    for(int i = 0; i < n; i++){
        printf("\n  [%d/%d] %s\n", i+1, n, cases[i].label);
        if(!validate(&cases[i], i+1)) continue;
        simulate(&cases[i], &results[i], fp_conv, valid_count);
        valid_cfgs[valid_count] = cases[i];
        valid_ms  [valid_count] = results[i];
        valid_count++;
    }

    if (fp_conv){ fclose(fp_conv); printf("\n  Conv stats saved to : %s\n", conv_file); }

    if(valid_count == 0){
        fprintf(stderr,"[ERROR] All cases failed validation.\n");
        return EXIT_FAILURE;
    }

    metrics_print_terminal(valid_count, valid_cfgs, valid_ms);

    FILE *fp = fopen(csv_file, "w");
    if(!fp){
        fprintf(stderr,"[WARN] Cannot write to '%s'.\n", csv_file);
    } else {
        metrics_write_csv(fp, valid_count, valid_cfgs, valid_ms);
        fclose(fp);
        printf("  Metrics CSV saved to : %s\n\n", csv_file);
    }

    return EXIT_SUCCESS;
}