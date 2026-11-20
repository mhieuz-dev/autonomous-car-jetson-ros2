"""Tham số của chương trình lái tự hành, gom một chỗ để chỉnh trên xe cho nhanh.

Bộ dò vạch + PID khoảng cách: core/lane_follow.py. Nhận hầm + luật lái trong hầm: core/tunnel.py.
Số nào ghi "(chưa chỉnh trên xe)" là đặt theo sa bàn giả, chạy thật phải chỉnh lại.
"""
from core import settings

# ===== THAM SỐ (sửa, Ctrl+S, chạy lại) =====================================
HZ = 20
MAX_RUN_TIME = 600.0       # s: chạy quá lâu thì tự dừng (phòng khi kẹt ở đâu đó mà không ai bấm dừng)

# --- bám vạch: hành vi mặc định khi không có gì khác cần lo ---
V_LANE = 0.35           # m/s trên đường thẳng, tự giảm theo sai số và độ cong (chưa chỉnh trên xe)
SEARCH_W = 0.6          # rad/s: tốc độ xoay phải mỗi nhịp tìm vạch
SEARCH_TURN_S = 0.25    # xoay bấy nhiêu giây (~8,6 độ ở 0.6 rad/s) ...
SEARCH_PAUSE_S = 0.5    # ... rồi đứng yên bấy nhiêu giây cho camera nhìn rõ, lặp tới khi thấy vạch
SEARCH_MAX_DEG = 360.0  # xoay tổng quá số độ này vẫn chưa thấy vạch = hết đường, dừng hẳn
# Nhảy sang vạch khác: vạch mới lệch > JUMP_MAX_M so với vạch đang bám VÀ |khoảng cách - D_TARGET| > PLAUSIBLE_ERR_M
# -> đứng yên, xoay phải từng nhịp tới khi thấy vạch phải hợp lý (cách tâm xe RIGHT_LINE_NEAR_M..XA_M) N_FRAMES_OK khung liền.
JUMP_CHECK_ON = True
JUMP_MAX_M = 0.12              # làn 0.40, vạch làn bên cách ~0.40
PLAUSIBLE_ERR_M = 0.12
RIGHT_LINE_NEAR_M = 0.10
RIGHT_LINE_FAR_M = 0.30
N_FRAMES_OK = 2

# --- người đi bộ (LiDAR phía trước) ---
LIDAR_TO_BUMPER_M = settings.LIDAR_TO_BUMPER_M  # sửa trong core/settings.py (đo bằng thước)
CLEARANCE_M = 0.30                              # dừng khi người cách CẢN TRƯỚC nhỏ hơn số này
REACTION_S = 0.25                               # s: cộng thêm V_LANE * số này vào ngưỡng (xe cần thời gian dừng)
HALF_WIDTH_M = 0.15                             # chỉ vật trong dải ±HALF_WIDTH_M trước mũi mới tính là người
PED_MIN_POINTS = 3                              # ít nhất bấy nhiêu điểm LiDAR trong vùng mới tính là có người
CLEAR_TIME_S = 1.0                              # vùng trước trống liên tục bao lâu thì chạy lại (s)

# --- đèn + biển (YOLO) ---
LIGHT_SOURCE = 'car'               # 'car' = model chạy trên Jetson | 'laptop' = laptop gửi kết quả qua DDS (tools/detector_laptop.py)
LIGHTS_ON = True
LIGHT_MIN_HEIGHT_PX = 40  # đèn 'đủ gần' khi khung bao cao >= số này (px, ảnh 640x480). Chọn bằng onnx_test
LIGHT_MIN_CONF = 0.5      # độ tin tối thiểu của đèn xanh
RED_MIN_CONF = 0.75       # riêng đèn ĐỎ / VÀNG (cao hơn = bớt dừng nhầm)
LIGHT_PERIOD_S = 0.5      # giây giữa 2 lần nhận đèn trên xe
GREEN_STREAK = 2          # số lần nhận XANH liên tiếp mới đi tiếp sau khi dừng
LIGHT_MAX_WAIT_S = 20.0   # chờ đèn tối đa (s); vẫn chưa xanh thì đi tiếp (tránh kẹt vì nhận nhầm) ...
LIGHT_IGNORE_S = 5.0      # ... và bỏ qua đèn bấy nhiêu giây để chạy qua khỏi đèn
SIGNS_ON = True
SIGN_MIN_CONF = 0.6
STOP_MIN_HEIGHT_PX = 60  # biển STOP 'đủ gần' khi khung cao >= số này (px)
STOP_WAIT_S = 1.0        # đứng yên bấy nhiêu giây ở biển STOP
STOP_IGNORE_S = 6.0      # dừng xong thì bỏ qua STOP bấy nhiêu giây (để xe chạy qua biển)

# --- dốc (IMU) ---
# Xe ngóc mũi khi chạy nhanh: sàn phẳng 0,30 m/s đã đo pitch +2,4..+4,0 độ (log 10/10), khớp NOSE_UP_DEG * v^2.
# Ngưỡng dốc dưới đây tính trên pitch ĐÃ TRỪ phần này, nên đổi tốc độ không làm sàn phẳng thành dốc.
NOSE_UP_DEG = 30.0
NOSE_UP_LAG_S = 0.3      # pitch đo được trễ sau tốc độ (thân xe + lọc bù IMU): lọc v^2 cùng độ trễ khi trừ
SLOPE_MIN_DIST_M = 0.20  # chạy ít nhất bấy nhiêu mét từ lúc xuất phát rồi mới xét dốc (rung lúc tăng tốc)
SLOPE_UP_DEG = 2.0       # pitch > số này liên tục SLOPE_UP_S = bắt đầu lên dốc (dốc thật +3,6 độ)
SLOPE_UP_S = 0.8
CREST_DEG = 1.0              # đang lên mà pitch < số này liên tục CREST_S = tới đỉnh
CREST_S = 0.6
SLOPE_DOWN_DEG = -2.2            # pitch < số này liên tục SLOPE_DOWN_S = xuống dốc (dốc thật -3,1; phẳng ở 0,15 m/s -1,9..+1,5)
SLOPE_DOWN_S = 0.5
SLOPE_END_DEG = -1.0              # đang xuống mà pitch > số này liên tục SLOPE_END_S = hết dốc
SLOPE_END_S = 0.5
CREST_MAX_LEN_M = 1.5  # đỉnh dài hơn chừng này mà chưa xuống thì coi như sàn phẳng
V_UPHILL = 0.30        # m/s (luật: không đứng yên, không lùi trên dốc)
V_CREST = 0.25
V_DOWNHILL = 0.15
HEADING_DEV_DEG = 8.0            # lệch hướng hơn chừng này trên dốc thì chậm lại chỉnh hướng trước
V_REALIGN = 0.12
KP_SLOPE_HEADING = 1.8
KI_SLOPE_HEADING = 0.3
KD_SLOPE_HEADING = 0.10
W_MAX_SLOPE = 0.8

# --- hầm (tốc độ + luật lái ở core/tunnel.py) ---
UNSTICK_ON = False     # tắt mặc định: hầm chạy đúng bản 04a. True = kẹt thì lùi ra rồi đi tiếp
UNSTICK_AFTER_S = 1.5  # bị chặn mũi / ra lệnh chạy mà bánh không lăn quá bấy nhiêu giây ...
REVERSE_V = -0.08      # ... thì lùi THẲNG với tốc độ này
REVERSE_DIST_M = 0.12  # lùi bấy nhiêu mét rồi quét lại, đi tiếp
REAR_CLEAR_M = 0.08    # dừng lùi sớm nếu sau cản sau còn ít hơn bấy nhiêu mét

# --- vừa ra hầm ---
V_EXIT_TUNNEL = 0.15
EXIT_COAST_MIN_S = 0.5  # giữ hướng lúc ra hầm ít nhất bấy nhiêu giây (chưa chỉnh trên xe; bản thi trôi đúng 2 s) ...
EXIT_COAST_MAX_S = 2.0  # ... tối đa bấy nhiêu giây
EXIT_LINE_FRAMES = 5    # thấy vạch bấy nhiêu khung liền thì trả lái cho bám vạch
# ===========================================================================
