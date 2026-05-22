/*
 * conv_simulator — Standard Convolution Architecture Simulator
 * ============================================================
 * Usage:  ./conv_sim [config_file]   (default: configs.txt)
 *
 * Reads one or more simulation cases from a config file, runs
 * three-stage simulation for each (Tiling → Spatial → Stationary),
 * then prints Table A & Table B to both stdout and results.txt.
 */

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

/* ── Validate tile sizes against problem dimensions ── */
static int validate(const Config *c, int case_num){
    int OH = out_dim(c->H, c->R, c->padding, c->stride);
    int OW = out_dim(c->W, c->S, c->padding, c->stride);
    int ok = 1;
    if(OH <= 0 || OW <= 0){
        fprintf(stderr,"[ERROR] Case %d: output dimension non-positive "
                "(OH=%d OW=%d). Check R,S,padding,stride.\n",
                case_num, OH, OW);
        ok = 0;
    }
    if(c->TH > OH || c->TW > OW){
        fprintf(stderr,"[WARN]  Case %d: tile (TH=%d,TW=%d) larger than "
                "output (%d,%d). Clamped automatically.\n",
                case_num, c->TH, c->TW, OH, OW);
    }
    if(c->TC > c->C || c->TK > c->K){
        fprintf(stderr,"[WARN]  Case %d: tile (TC=%d,TK=%d) larger than "
                "channels (C=%d,K=%d). Clamped automatically.\n",
                case_num, c->TC, c->TK, c->C, c->K);
    }
    return ok;
}

/* ── Run all three simulation stages for one case ── */
static void simulate(const Config *cfg, Metrics *m){
    metrics_reset(m);
    run_tiling    (cfg, m);   /* Stage 1: DRAM ↔ SRAM */
    run_spatial   (cfg, m);   /* Stage 2: SRAM → PE   */
    run_stationary(cfg, m);   /* Stage 3: Temporal reuse */
}

/* ── Print per-case progress to terminal ── */
static void print_progress(int i, int n, const Config *c){
    printf("  [%d/%d] %-12s  TH=%d TW=%d TC=%d TK=%d  R=%d S=%d  "
           "stride=%d pad=%d  strategy=%s\n",
           i, n, c->label,
           c->TH, c->TW, c->TC, c->TK,
           c->R,  c->S,
           c->stride, c->padding,
           strat_str(c->strategy));
}

int main(int argc, char *argv[])
{
    const char *cfg_file  = (argc > 1) ? argv[1] : "configs.txt";
    const char *out_file  = (argc > 2) ? argv[2] : "results.txt";

    printf("\n================================================================================\n");
    printf("              CONVOLUTION ARCHITECTURE SIMULATOR\n");
    printf("================================================================================\n");
    printf("  Config file : %s\n", cfg_file);
    printf("  Output file : %s\n\n", out_file);

    /* ── 1. Parse config file ── */
    Config  cases[MAX_CASES];
    int n = parse_config(cfg_file, cases);
    if(n == 0){
        fprintf(stderr,"[ERROR] No cases to simulate. Exiting.\n");
        return EXIT_FAILURE;
    }
    printf("  Found %d case(s). Simulating...\n\n", n);

    /* ── 2. Simulate each case ── */
    Metrics results[MAX_CASES];
    int valid_count = 0;

    /* We'll collect only valid cases for reporting */
    Config  valid_cfgs[MAX_CASES];
    Metrics valid_ms  [MAX_CASES];

    for(int i = 0; i < n; i++){
        print_progress(i+1, n, &cases[i]);
        if(!validate(&cases[i], i+1)) continue;
        simulate(&cases[i], &results[i]);
        valid_cfgs[valid_count] = cases[i];
        valid_ms  [valid_count] = results[i];
        valid_count++;
    }

    if(valid_count == 0){
        fprintf(stderr,"[ERROR] All cases failed validation.\n");
        return EXIT_FAILURE;
    }

    printf("\n  Simulation complete — %d case(s) processed.\n", valid_count);

    /* ── 3. Print report to stdout ── */
    printf("\n");
    metrics_print_report(stdout, valid_count, valid_cfgs, valid_ms);

    /* ── 4. Write report to results.txt ── */
    FILE *fp = fopen(out_file, "w");
    if(!fp){
        fprintf(stderr,"[WARN] Cannot write to '%s'. Results not saved.\n", out_file);
    } else {
        metrics_print_report(fp, valid_count, valid_cfgs, valid_ms);
        fclose(fp);
        printf("  Results saved to: %s\n\n", out_file);
    }

    return EXIT_SUCCESS;
}
