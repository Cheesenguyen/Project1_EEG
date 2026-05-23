#include "spatial.h"
#include "utils.h"

/*
 * Spatial Unrolling — SRAM → PE array
 * 
 * Chiến lược: unroll theo chiều TK (output channels)
 * .
 *   → Mỗi cycle, TK_eff PE được kích hoạt song song.
 *   → Mỗi PE_k tính: out[oh,ow,k] += in[ih,iw,c] * w[k,c,r,s]
 *
 * Loop thực tế bên trong mỗi tile (th,tw,tc,tk):
 *   for oh in TH_eff
 *     for ow in TW_eff
 *       for r in R
 *         for s in S
 *           for c in TC_eff
 *             [PARALLEL over k in TK_eff]   ← 1 cycle
 *
 * Broadcast / Read policy:
 *   Input  : in[ih,iw,c] BROADCAST tới tất cả TK_eff PE
 *            → tất cả PE dùng CÙNG 1 giá trị → 1 địa chỉ SRAM duy nhất
 *            → sram_input_reads  += 1 / cycle
 *   Weight : mỗi PE_k đọc w[k,c,r,s] với k KHÁC NHAU → TK_eff địa chỉ khác nhau
 *            → dù đọc song song vẫn là TK_eff lần truy cập Weight buffer
 *            → sram_weight_reads += TK_eff / cycle
 *
 * Tổng:
 *   cycles_per_tile   = TH_eff × TW_eff × R × S × TC_eff
 *   total_cycles      = cycles_per_tile × n_th × n_tw × n_tc × n_tk
 *   sram_input_reads  = total_cycles × 1        (broadcast — 1 địa chỉ)
 *   sram_weight_reads = total_cycles × TK_eff   (mỗi PE đọc weight riêng)
 *
 * Lưu ý: max_PE = TK_eff (số PE kích hoạt song song mỗi cycle)
 */
void run_spatial(const Config *cfg, Metrics *m)
{
    // tính kích thước đầu ra
    int OH = out_dim(cfg->H, cfg->R, cfg->padding, cfg->stride);
    int OW = out_dim(cfg->W, cfg->S, cfg->padding, cfg->stride);

    // tổng số tile phải cắt theo các chiều khác nhau. Vdu ảnh OH = 100, TH = 32 => n_th = 4
    int n_th = CEIL_DIV(OH, cfg->TH);
    int n_tw = CEIL_DIV(OW, cfg->TW);
    int n_tc = CEIL_DIV(cfg->C, cfg->TC);
    int n_tk = CEIL_DIV(cfg->K, cfg->TK);

    // lấy kích thước thực tế của phần tile, tránh cấp phát thừa
    int TH_eff = MIN(cfg->TH, OH);
    int TW_eff = MIN(cfg->TW, OW);
    int TC_eff = MIN(cfg->TC, cfg->C);
    int TK_eff = MIN(cfg->TK, cfg->K);

    // số lượng PE tính song song trong 1 lần = số lượng output channel
    m->max_PE = TK_eff;

    // số chu kỳ cần thực hiên với mỗi tile (th * tw* r * r * c) do unroll theo k nên không còn k
    long cycles_per_tile = (long)TH_eff * TW_eff * cfg->R * cfg->S * TC_eff;
    long n_tiles         = (long)n_th * n_tw * n_tc * n_tk;

    m->total_cycles = cycles_per_tile * n_tiles;

    /*
     * số lần đọc dữ liệu từ sram trong 1 chu kỳ:
     * input buffer: 1 lần đọc -> dùng chung cho toàn bộ PE
     * weight buffer: TK_eff giá trị trọng số <=> TK_eff lần đọc
     */
    m->sram_input_reads  = m->total_cycles;            // 1 lần đọc * số chu kỳ
    m->sram_weight_reads = m->total_cycles * TK_eff;   // TK-eff lần đọc * số chu kỳ
