#include "tiling.h"
#include "utils.h"

/*
 * Tiling — DRAM -> SRAM
 *
 * Vòng lặp tile (thứ tự ngoài → trong):
 *   for th  in CEIL(OH/TH)
 *     for tw  in CEIL(OW/TW)
 *       for tk  in CEIL(K/TK)      ← output-channel tile
 *         for tc  in CEIL(C/TC)    ← input-channel  tile
 *
 * DRAM access policy:
 *   Input tile  (th,tw,tc) : cần cho tất cả tk → load 1 lần / (th,tw,tc)
 *                             = n_th × n_tw × n_tc  lần load
 *   Weight tile (tk,tc)    : dùng cho tất cả (th,tw) → load 1 lần / (tk,tc)
 *                             = n_tk × n_tc  lần load
 *   Output tile (th,tw,tk) : ghi sau khi tc loop xong
 *                             = n_th × n_tw × n_tk  lần store
 *
 * SRAM sizing (float32 = 4 bytes/element):
 *   Input  buffer : receptive-field footprint = (TH_eff-1)*stride+R
 *                                             × (TW_eff-1)*stride+S
 *                                             × TC_eff
 *   Weight buffer : TK_eff × TC_eff × R × S
 *   Psum buffer   : TH_eff × TW_eff × TK_eff   (output tile)
 */
void run_tiling(const Config *cfg, Metrics *m)
{
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

    // tính ngược lại kích thước của input 
    int rf_h = (TH_eff - 1) * cfg->stride + cfg->R;
    int rf_w = (TW_eff - 1) * cfg->stride + cfg->S;

    // Dung lượng đã sử dụng trong sram
    m->sram_input_size  = (long)rf_h  * rf_w  * TC_eff  * 4;
    m->sram_weight_size = (long)TK_eff * TC_eff * cfg->R * cfg->S * 4;
    m->sram_psum_size   = (long)TH_eff * TW_eff * TK_eff * 4;
    m->sram_total_size   = m->sram_input_size + m->sram_weight_size + m->sram_psum_size;

    // đếm số lần load từng biến
    long input_loads   = (long)n_th * n_tw * n_tc;
    long weight_loads  = (long)n_tk * n_tc;
    long output_stores = (long)n_th * n_tw * n_tk;

    m->dram_input_loads  = input_loads;
    m->dram_weight_loads = weight_loads;
    m->dram_loads        = input_loads + weight_loads;
    m->dram_stores = output_stores;
}
