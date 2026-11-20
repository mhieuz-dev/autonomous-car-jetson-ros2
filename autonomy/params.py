"""Tham số của chương trình lái tự hành, gom một chỗ để chỉnh trên xe cho nhanh.

Bộ dò vạch + PID khoảng cách: core/lane_follow.py. Nhận hầm + luật lái trong hầm: core/tunnel.py.
Số nào ghi "(chưa chỉnh trên xe)" là đặt theo sa bàn giả, chạy thật phải chỉnh lại.
"""
from core import settings

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
HZ = 20
THOI_GIAN_TOI_DA = 600.0       # s: chạy quá lâu thì tự dừng (phòng khi kẹt ở đâu đó mà không ai bấm dừng)

# --- bám vạch: hành vi mặc định khi không có gì khác cần lo ---
V_BAM_VACH = 0.35              # m/s trên đường thẳng, tự giảm theo sai số và độ cong (chưa chỉnh trên xe)
TIM_W = 0.6                    # rad/s: tốc độ xoay phải mỗi nhịp tìm vạch
TIM_XOAY_GIAY = 0.25           # xoay bấy nhiêu giây (~8,6 độ ở 0.6 rad/s) ...
TIM_DUNG_GIAY = 0.5            # ... rồi đứng yên bấy nhiêu giây cho camera nhìn rõ, lặp tới khi thấy vạch
TIM_GOC_MAX_DO = 360.0         # xoay tổng quá số độ này vẫn chưa thấy vạch = hết đường, dừng hẳn
# Nhảy sang vạch khác: vạch mới lệch > NHAY_MAX_M so với vạch đang bám VÀ |khoảng cách - D_DICH| > SAI_SO_BAT_HOP_LY_M
# -> đứng yên, xoay phải từng nhịp tới khi thấy vạch phải hợp lý (cách tâm xe VACH_PHAI_GAN_M..XA_M) N_KHUNG_OK khung liền.
NHAY_BAT = True
NHAY_MAX_M = 0.12              # làn 0.40, vạch làn bên cách ~0.40
SAI_SO_BAT_HOP_LY_M = 0.12
VACH_PHAI_GAN_M = 0.10
VACH_PHAI_XA_M = 0.30
N_KHUNG_OK = 2

# --- người đi bộ (LiDAR phía trước) ---
LIDAR_TOI_CAN_M = settings.LIDAR_TOI_CAN_M   # sửa trong core/settings.py (đo bằng thước)
KHOANG_HO_M = 0.30             # dừng khi người cách CẢN TRƯỚC nhỏ hơn số này
THOI_GIAN_PHAN_UNG = 0.25      # s: cộng thêm V_BAM_VACH * số này vào ngưỡng (xe cần thời gian dừng)
NUA_RONG_M = 0.15              # chỉ vật trong dải ±NUA_RONG_M trước mũi mới tính là người
SO_DIEM_MIN = 3                # ít nhất bấy nhiêu điểm LiDAR trong vùng mới tính là có người
THOI_GIAN_TRONG = 1.0          # vùng trước trống liên tục bao lâu thì chạy lại (s)

# --- đèn + biển (YOLO) ---
DEN_NGUON = 'xe'               # 'xe' = model chạy trên Jetson | 'laptop' = laptop gửi kết quả qua DDS (tools/detector_laptop.py)
DEN_BAT = True
DEN_CAO_MIN_PX = 40            # đèn 'đủ gần' khi khung bao cao >= số này (px, ảnh 640x480). Chọn bằng onnx_test
DEN_TIN_MIN = 0.5              # độ tin tối thiểu của đèn xanh
DEN_DO_TIN_MIN = 0.75          # riêng đèn ĐỎ / VÀNG (cao hơn = bớt dừng nhầm)
DEN_CHU_KY = 0.5               # giây giữa 2 lần nhận đèn trên xe
DEN_XANH_LIEN = 2              # số lần nhận XANH liên tiếp mới đi tiếp sau khi dừng
DEN_CHO_TOI_DA = 20.0          # chờ đèn tối đa (s); vẫn chưa xanh thì đi tiếp (tránh kẹt vì nhận nhầm) ...
DEN_BO_QUA_GIAY = 5.0          # ... và bỏ qua đèn bấy nhiêu giây để chạy qua khỏi đèn
BIEN_BAT = True
BIEN_TIN_MIN = 0.6
STOP_CAO_MIN_PX = 60           # biển STOP 'đủ gần' khi khung cao >= số này (px)
STOP_DUNG_GIAY = 1.0           # đứng yên bấy nhiêu giây ở biển STOP
STOP_BO_QUA_GIAY = 6.0         # dừng xong thì bỏ qua STOP bấy nhiêu giây (để xe chạy qua biển)

# --- dốc (IMU) ---
# Xe ngóc mũi khi chạy nhanh: sàn phẳng 0,30 m/s đã đo pitch +2,4..+4,0 độ (log 10/10), khớp NGOC_MUI_DO * v^2.
# Ngưỡng dốc dưới đây tính trên pitch ĐÃ TRỪ phần này, nên đổi tốc độ không làm sàn phẳng thành dốc.
NGOC_MUI_DO = 30.0
NGOC_MUI_TRE_S = 0.3            # pitch đo được trễ sau tốc độ (thân xe + lọc bù IMU): lọc v^2 cùng độ trễ khi trừ
DOC_QUANG_DUONG_MIN = 0.20     # chạy ít nhất bấy nhiêu mét từ lúc xuất phát rồi mới xét dốc (rung lúc tăng tốc)
DOC_LEN_DO = 2.0               # pitch > số này liên tục DOC_LEN_GIAY = bắt đầu lên dốc (dốc thật +3,6 độ)
DOC_LEN_GIAY = 0.8
DOC_DINH_DO = 1.0              # đang lên mà pitch < số này liên tục DOC_DINH_GIAY = tới đỉnh
DOC_DINH_GIAY = 0.6
DOC_XUONG_DO = -2.2            # pitch < số này liên tục DOC_XUONG_GIAY = xuống dốc (dốc thật -3,1; phẳng ở 0,15 m/s -1,9..+1,5)
DOC_XUONG_GIAY = 0.5
DOC_HET_DO = -1.0              # đang xuống mà pitch > số này liên tục DOC_HET_GIAY = hết dốc
DOC_HET_GIAY = 0.5
DINH_DAI_TOI_DA_M = 1.5        # đỉnh dài hơn chừng này mà chưa xuống thì coi như sàn phẳng
V_LEN_DOC = 0.30               # m/s (luật: không đứng yên, không lùi trên dốc)
V_DINH_DOC = 0.25
V_XUONG_DOC = 0.15
LECH_NHIEU_DO = 8.0            # lệch hướng hơn chừng này trên dốc thì chậm lại chỉnh hướng trước
V_CHINH_HUONG = 0.12
KP_HUONG_DOC = 1.8
KI_HUONG_DOC = 0.3
KD_HUONG_DOC = 0.10
W_MAX_DOC = 0.8

# --- hầm (tốc độ + luật lái ở core/tunnel.py) ---
GO_KET_BAT = False             # tắt mặc định: hầm chạy đúng bản 04a. True = kẹt thì lùi ra rồi đi tiếp
GO_KET_SAU = 1.5               # bị chặn mũi / ra lệnh chạy mà bánh không lăn quá bấy nhiêu giây ...
LUI_V = -0.08                  # ... thì lùi THẲNG với tốc độ này
LUI_QUANG_DUONG = 0.12         # lùi bấy nhiêu mét rồi quét lại, đi tiếp
SAU_TRONG_M = 0.08             # dừng lùi sớm nếu sau cản sau còn ít hơn bấy nhiêu mét

# --- vừa ra hầm ---
V_RA_HAM = 0.15
RA_HAM_TROI_MIN = 0.5          # giữ hướng lúc ra hầm ít nhất bấy nhiêu giây (chưa chỉnh trên xe; bản thi trôi đúng 2 s) ...
RA_HAM_TROI_MAX = 2.0          # ... tối đa bấy nhiêu giây
RA_HAM_SO_KHUNG_VACH = 5       # thấy vạch bấy nhiêu khung liền thì trả lái cho bám vạch
# ===========================================================================
