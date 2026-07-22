#include "stationary.h"
#include "utils.h"
#include <string.h>

// xóa dữ liệu trong PE để tính ps mới
// nhận *ps_buf, td chưa thông số cấu hình của khối tile hiện tại
void ps_buf_init(float *ps_buf, const TileDesc *td)
{
    long sz = (long)td->TH_eff * td->TW_eff * td->TK_eff;   // tổng số lượng ptu cần cấp phát
    memset(ps_buf, 0, sz * sizeof(float)); // đưa về 0
}

//đọc sram gih vào thanh ghi
void ps_buf_load(float *ps_buf, const float *ps_store, const TileDesc *td)
 // ps_store con trỏ chri vào vùng nhớ SRAM DRAM
 // ps_buf: thanh ghi bên trong PE
{
    // tính tổng số lượng điểm ảnh của khối tile 
    long sz = (long)td->TH_eff * td->TW_eff * td->TK_eff;
    memcpy(ps_buf, ps_store, sz * sizeof(float));       // copy dữ liệu từ ps_store, ps_buf
}

// ghi dữ liệu từ PE về DRAM
void ps_buf_save(const float *ps_buf, float *ps_store, const TileDesc *td)
{
    long sz = (long)td->TH_eff * td->TW_eff * td->TK_eff;       // xác định kích thước vùng nhớ cần đẩy ra
    memcpy(ps_store, ps_buf, sz * sizeof(float));   // dán dữ liệu từ thanh ghi ra ngoài 
}