"""
verify.py — Script tự động kiểm tra chéo kết quả Convolution (C vs Python)
Yêu cầu: numpy, pandas, scipy
Cách dùng: 
    python verify.py [file_config.csv] [file_stats_tu_C.csv]
Ví dụ: 
    python verify.py results.csv conv_stats.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy.signal import correlate2d

def out_dim(in_size, filt, pad, stride):
    return (in_size + 2 * pad - filt) // stride + 1

def lcg_fill(n, seed=42):
    """Tái tạo chính xác bộ sinh số giả ngẫu nhiên từ code C"""
    seed = seed & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = (seed & 0xFFFF) / 65535.0 - 0.5
    return out, seed

def conv_forward(H, W, C_in, K, R, S, stride, padding):
    """Tính toán Convolution tham chiếu bằng thư viện Scipy"""
    OH = out_dim(H, R, padding, stride)
    OW = out_dim(W, S, padding, stride)

    inp_flat, seed = lcg_fill(H * W * C_in, seed=42)
    wt_flat, _ = lcg_fill(K * R * S * C_in, seed=seed)

    inp = inp_flat.reshape(H, W, C_in)
    wt = wt_flat.reshape(K, R, S, C_in)

    output = np.zeros((OH, OW, K), dtype=np.float32)

    for k in range(K):
        for c in range(C_in):
            feat_map = inp[:, :, c]
            kernel = wt[k, :, :, c]
            full = correlate2d(feat_map, kernel, mode='full')
            
            r0 = R - 1 - padding
            c0 = S - 1 - padding
            
            for oh in range(OH):
                for ow in range(OW):
                    output[oh, ow, k] += full[r0 + oh * stride, c0 + ow * stride]

    return output, OH, OW

def main():
    # Nhận tên file từ argument dòng lệnh, hoặc dùng mặc định
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    stats_file = sys.argv[2] if len(sys.argv) > 2 else "conv_stats.csv"

    try:
        df_config = pd.read_csv(csv_file)
        df_stats = pd.read_csv(stats_file)
    except FileNotFoundError as e:
        print(f"[LỖI] Không tìm thấy file: {e.filename}")
        print("Hãy đảm bảo bạn đã chạy file thực thi C trước khi chạy kịch bản này.")
        sys.exit(1)

    print("  TỰ ĐỘNG ĐỐI CHIẾU KẾT QUẢ CONVOLUTION (C vs PYTHON)")

    # Chuyển df_stats thành dictionary để tra cứu nhanh: dict[(label, channel_k)] = row
    c_stats_dict = {}
    for _, row in df_stats.iterrows():
        c_stats_dict[(row['label'], int(row['channel_k']))] = row

    all_passed = True
    TOLERANCE = 5e-2  # Ngưỡng sai số chấp nhận được cho float32

    for _, row in df_config.iterrows():
        H, W, C_in = int(row['H']), int(row['W']), int(row['C'])
        K, R, S = int(row['K']), int(row['R']), int(row['S'])
        stride, padding = int(row['stride']), int(row['padding'])
        label = str(row['label'])

        print(f"Đang kiểm tra: {label} (In: {H}x{W}x{C_in}, K={K}) ...", end=" ")

        # 1. Chạy hàm Python để lấy kết quả tham chiếu
        out_ref, OH, OW = conv_forward(H, W, C_in, K, R, S, stride, padding)
        
        case_passed = True
        
        # 2. Lặp qua từng Output Channel để đối chiếu thống kê
        for k in range(K):
            key = (label, k)
            if key not in c_stats_dict:
                print(f"\n  [CẢNH BÁO] Không tìm thấy thống kê của C cho {label}, channel k={k}")
                case_passed = False
                continue
                
            c_row = c_stats_dict[key]
            ch_data = out_ref[:, :, k]
            
            p_min, p_max = ch_data.min(), ch_data.max()
            p_mean, p_sum = ch_data.mean(), ch_data.sum()
            
            c_min, c_max = float(c_row['min']), float(c_row['max'])
            c_mean, c_sum = float(c_row['mean']), float(c_row['sum'])

            # 3. So sánh với ngưỡng TOLERANCE
            tol_min_max = 5e-2  # Ngưỡng cho min, max (chấp nhận lệch 0.05)
            tol_mean    = 5e-2  # Ngưỡng cho giá trị trung bình
            tol_sum     = 5.0   # Ngưỡng cho tổng (nới lỏng hẳn ra 5.0 vì cộng dồn hàng vạn số)

            if not (np.isclose(p_min, c_min, atol=tol_min_max) and 
                    np.isclose(p_max, c_max, atol=tol_min_max) and 
                    np.isclose(p_mean, c_mean, atol=tol_mean) and 
                    np.isclose(p_sum, c_sum, atol=tol_sum)):
                print(f"\n  [LỖI] Kênh k={k} không khớp dữ liệu!")
                print(f"    Python : min={p_min:.6f}, max={p_max:.6f}, mean={p_mean:.6f}, sum={p_sum:.6f}")
                print(f"    C      : min={c_min:.6f}, max={c_max:.6f}, mean={c_mean:.6f}, sum={c_sum:.6f}")
                case_passed = False
                all_passed = False

        if case_passed:
            print("PASS")

    print("\n" + "=" * 65)
    if all_passed:
        print("  KẾT LUẬN: TOÀN BỘ CÁC LỚP ĐỀU KHỚP KẾT QUẢ TÍNH TOÁN (PASS)")
    else:
        print("  KẾT LUẬN: PHÁT HIỆN LỖI SAI LỆCH DỮ LIỆU (FAIL)")
    print("=" * 65)

if __name__ == "__main__":
    main()