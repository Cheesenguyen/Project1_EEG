#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "parser.h"
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

static void simulate(const Config *cfg, Metrics *m, FILE *fp_conv,
                     int case_idx, int unroll_mode)
{
    float *inp = NULL, *wt = NULL, *out = NULL;
    int OH, OW;

    /* conv_forward vừa chạy thật vừa đếm metrics trực tiếp */
    if (conv_forward(cfg, m, &inp, &wt, &out, &OH, &OW, unroll_mode) == 0) {
        conv_print_sample(out, OH, OW, cfg->K, cfg->label);
        if (fp_conv && unroll_mode == 0)
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
    const char *csv_tk    = (argc > 2) ? argv[2] : "results_tk.csv";
    const char *csv_thw   = (argc > 3) ? argv[3] : "results_thw.csv";
    const char *conv_file = (argc > 4) ? argv[4] : "conv_stats.csv";

    printf("              CONVOLUTION ARCHITECTURE SIMULATOR\n");
    printf("  Config file  : %s\n", cfg_file);
    printf("  Metrics TK   : %s\n", csv_tk);
    printf("  Metrics THW  : %s\n", csv_thw);
    printf("  Conv stats   : %s\n\n", conv_file);

    Config cases[MAX_CASES];
    int n = parse_config(cfg_file, cases);
    if(n == 0){
        fprintf(stderr,"[ERROR] No cases to simulate.\n");
        return EXIT_FAILURE;
    }
    printf("  Found %d case(s). Simulating...\n", n);

    FILE *fp_conv = fopen(conv_file, "w");
    if (!fp_conv)
        fprintf(stderr,"[WARN] Cannot open '%s' for conv stats.\n", conv_file);

    Config  valid_cfgs[MAX_CASES];
    Metrics ms_tk     [MAX_CASES];
    Metrics ms_thw    [MAX_CASES];
    int valid_count = 0;

    for(int i = 0; i < n; i++){
        printf("\n  [%d/%d] %s\n", i+1, n, cases[i].label);
        if(!validate(&cases[i], i+1)) continue;

        simulate(&cases[i], &ms_tk [valid_count], fp_conv, valid_count, /*TK*/  0);
        simulate(&cases[i], &ms_thw[valid_count], fp_conv, valid_count, /*THW*/ 1);

        valid_cfgs[valid_count] = cases[i];
        valid_count++;
    }

    if (fp_conv){ fclose(fp_conv); printf("\n  Conv stats saved to : %s\n", conv_file); }

    if(valid_count == 0){
        fprintf(stderr,"[ERROR] All cases failed validation.\n");
        return EXIT_FAILURE;
    }

    printf("\n          UNROLL: TK \n");
    metrics_print_terminal(valid_count, valid_cfgs, ms_tk);

    printf("\n          UNROLL: TH×TW \n");
    metrics_print_terminal(valid_count, valid_cfgs, ms_thw);

    FILE *fp;
    fp = fopen(csv_tk, "w");
    if(!fp) fprintf(stderr,"[WARN] Cannot write '%s'.\n", csv_tk);
    else { metrics_write_csv(fp, valid_count, valid_cfgs, ms_tk,  "TK");  fclose(fp); }

    fp = fopen(csv_thw, "w");
    if(!fp) fprintf(stderr,"[WARN] Cannot write '%s'.\n", csv_thw);
    else { metrics_write_csv(fp, valid_count, valid_cfgs, ms_thw, "THW"); fclose(fp); }

    printf("  Metrics TK  saved to : %s\n", csv_tk);
    printf("  Metrics THW saved to : %s\n\n", csv_thw);

    return EXIT_SUCCESS;
}