#include "conv.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ── Simple LCG rand in [-0.5, 0.5] for reproducible fills ── */
static float lcg_rand(unsigned int *seed)
{
    *seed = (*seed) * 1664525u + 1013904223u;
    return (float)(*seed & 0xFFFF) / 65535.0f - 0.5f;
}

/*
 * conv_forward
 * ============
 * Tính phép tích chập chuẩn (standard 2D convolution, multi-channel).
 *
 * Layout:
 *   input  : [H][W][C]      → index: (h*W + w)*C + c
 *   weight : [K][R][S][C]   → index: ((k*R + r)*S + s)*C + c
 *   output : [OH][OW][K]    → index: (oh*OW + ow)*K + k
 *
 * Padding được xử lý bằng zero-padding (giá trị ngoài biên = 0).
 */
int conv_forward(const Config *cfg,
                 float **out_input,
                 float **out_weight,
                 float **out_output,
                 int *OH_out, int *OW_out)
{
    int OH = out_dim(cfg->H, cfg->R, cfg->padding, cfg->stride);
    int OW = out_dim(cfg->W, cfg->S, cfg->padding, cfg->stride);
    *OH_out = OH;
    *OW_out = OW;

    long input_sz  = (long)cfg->H * cfg->W * cfg->C;
    long weight_sz = (long)cfg->K * cfg->R * cfg->S * cfg->C;
    long output_sz = (long)OH * OW * cfg->K;

    float *input  = (float *)malloc(input_sz  * sizeof(float));
    float *weight = (float *)malloc(weight_sz * sizeof(float));
    float *output = (float *)calloc(output_sz,  sizeof(float)); /* zero-init for accumulation */

    if (!input || !weight || !output) {
        free(input); free(weight); free(output);
        return -1;
    }

    /* Fill input and weight with deterministic pseudo-random values */
    unsigned int seed = 42u;
    for (long i = 0; i < input_sz;  i++) input [i] = lcg_rand(&seed);
    for (long i = 0; i < weight_sz; i++) weight[i] = lcg_rand(&seed);

    /*
     * Core convolution loop
     * ---------------------
     * Tiling-aware loop order: iterate output positions, then filter, then channels.
     * Đây là "reference implementation" — không tối ưu, nhưng đúng về mặt toán học.
     */
    int H = cfg->H, W = cfg->W, C = cfg->C;
    int K = cfg->K, R = cfg->R, S = cfg->S;
    int stride = cfg->stride, padding = cfg->padding;

    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int k = 0; k < K; k++) {
                float acc = 0.0f;
                for (int r = 0; r < R; r++) {
                    int ih = oh * stride + r - padding;
                    if (ih < 0 || ih >= H) continue;   /* zero-pad */
                    for (int s = 0; s < S; s++) {
                        int iw = ow * stride + s - padding;
                        if (iw < 0 || iw >= W) continue; /* zero-pad */
                        for (int c = 0; c < C; c++) {
                            float in_val = input [(ih * W + iw) * C + c];
                            float wt_val = weight[((k * R + r) * S + s) * C + c];
                            acc += in_val * wt_val;
                        }
                    }
                }
                output[(oh * OW + ow) * K + k] = acc;
            }
        }
    }

    *out_input  = input;
    *out_weight = weight;
    *out_output = output;
    return 0;
}

/*
 * conv_write_stats
 * ================
 * Ghi thống kê min/max/mean/sum của từng output channel k vào file CSV.
 *
 * Mỗi dòng: label, k, min, max, mean, sum
 * append=0 → ghi header trước; append=1 → không ghi header (case tiếp theo)
 */
void conv_write_stats(FILE *fp, const float *output, int OH, int OW, int K,
                      const char *label, int append)
{
    if (!append)
        fprintf(fp, "label,channel_k,min,max,mean,sum\n");

    long spatial = (long)OH * OW;   /* số phần tử mỗi channel */

    for (int k = 0; k < K; k++) {
        float vmin = output[k], vmax = output[k], vsum = 0.0f;

        for (long i = 0; i < spatial; i++) {
            float v = output[i * K + k];   /* layout [OH][OW][K] */
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
            vsum += v;
        }
        float vmean = vsum / (float)spatial;

        fprintf(fp, "\"%s\",%d,%.6f,%.6f,%.6f,%.6f\n",
                label, k, vmin, vmax, vmean, vsum);
    }
}

/*
 * conv_print_sample
 * =================
 * In 3×3 góc trên-trái và vài thống kê cơ bản để kiểm tra kết quả.
 */
void conv_print_sample(const float *output, int OH, int OW, int K,
                       const char *label)
{
    printf("\n  [Conv output — %s]\n", label);

    /* Compute min/max/mean over first output channel (k=0) */
    float vmin = output[0], vmax = output[0], vsum = 0.0f;
    long  cnt  = (long)OH * OW;
    for (long i = 0; i < cnt; i++) {
        float v = output[i * K];   /* k=0 */
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        vsum += v;
    }
    printf("    Output shape : %d × %d × %d\n", OH, OW, K);
    printf("    Ch-0 stats   : min=%.4f  max=%.4f  mean=%.4f\n",
           vmin, vmax, vsum / cnt);

    /* Print top-left 3×3 corner, channel 0 */
    int rows = (OH < 3) ? OH : 3;
    int cols = (OW < 3) ? OW : 3;
    printf("    Top-left %dx%d corner (k=0):\n", rows, cols);
    for (int oh = 0; oh < rows; oh++) {
        printf("      ");
        for (int ow = 0; ow < cols; ow++) {
            printf("%10.4f ", output[(oh * OW + ow) * K + 0]);
        }
        printf("\n");
    }
}