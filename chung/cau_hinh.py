"""Cấu hình chung cho mọi script: đường dẫn, chủ đề ROS, cổng xem hình."""
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]

# Chủ đề ROS trên xe thật (đo 09/10: camera 30 Hz, /scan 10 Hz, /imu và /odom 25 Hz)
TOPIC_CAMERA = '/camera/image_raw/compressed'
TOPIC_SCAN = '/scan'
TOPIC_IMU = '/imu'
TOPIC_ODOM = '/odom'
TOPIC_LENH = '/cmd_vel'
TOPIC_DEN = '/champion/den'            # LAPTOP -> xe: kết quả nhận biển/đèn (JSON trong std_msgs/String), xem cong_cu/den_laptop.py

# Driver BTC đã xoay /scan 180 độ (CRC_SCAN_YAW_OFFSET_DEG=180 trong ~/crc_car.env).
# Nếu lidar_test cho thấy "trước" và "sau" bị ngược thì đặt 180 ở đây.
LIDAR_LECH_GOC_DO = 0.0

# Kích thước xe tính từ tâm LiDAR. ĐO LẠI BẰNG THƯỚC (số đang là tượng trưng)!
LIDAR_TOI_CAN_M = 0.197       # tâm LiDAR -> mép cản TRƯỚC
LIDAR_TOI_CAN_SAU_M = 0.069   # tâm LiDAR -> mép cản SAU

# CHẶN SÁT MŨI (mọi giai đoạn, không điều kiện): vật trong nón ±CHAN_MUI_NUA_GOC_DO
# cách cản trước < CHAN_MUI_CACH_M thì CẤM CHẠY TỚI (v = 0). Vẫn được quay tại chỗ
# và lùi, nên khúc cua hầm vẫn xoay ra được. Vật đi khỏi thì chạy tiếp.
CHAN_MUI_BAT = False   # TẮT mặc định (user chốt 10/10: giữ LiDAR 04a nguyên). Bật = True nếu muốn thử
CHAN_MUI_CACH_M = 0.05
CHAN_MUI_NUA_GOC_DO = 20.0

CONG_XEM = 8097                       # trang hình + trạng thái của script đang chạy
FILE_DUNG = Path('/tmp/champion_dung')  # có file này là mọi script dừng xe ngay

THU_MUC_LOG = GOC / 'logs'
CALIB = GOC / 'calib'                 # calib ĐANG DÙNG (commit lên git)
CALIB_MOI = GOC / 'calib_moi'         # calib vừa đo trên xe, chờ duyệt (`chay.sh nhan_calib`)
MODEL_ONNX = GOC / 'models' / 'best.onnx'
MODEL_ENGINE = GOC / 'models' / 'best.engine'

# Vòng gyro (chung cho mọi giai đoạn): bù cho bánh lệch / trượt bằng gyro z.
# Bản cũ chạy cái này trên xe (imu_yaw_loop.py, kp 0.5). Tắt = BAT_VONG_GYRO False.
BAT_VONG_GYRO = True
GYRO_KP = 0.5
GYRO_KI = 1.0
GYRO_I_MAX = 0.3

# Thứ tự các giai đoạn của FULL MAP
# Bản S7 dùng trong full map: 's7_bam_vach' (CHÍNH THỨC: bám vạch, giữ độ cong + cổng hợp lý; STOP đầu tiên thì chạy
# lại đường đẩy tay calib/duong_stop.csv rồi bám vạch V_BAM_SAU) | 's7_phat_lai' (bám vạch, mất vạch thì theo đường
# đã ghi) | 's7_ghi_duong' (ĐẨY TAY ghi đường từ đầu S7 tới đích, xe không tự chạy)
# | 's7_stop_ghi' (bám vạch, gặp STOP thì đẩy tay ghi đường cho s7_bam_vach chạy lại)
S7_DUNG = 's7_bam_vach'
# S0 (chờ đèn xanh) đã bỏ khỏi luật thi: full map bắt đầu từ S1, S0 vẫn chạy riêng được
THU_TU = ['s1_xuat_phat', 's2_len_doc', 's3_dinh_doc',
          's4_xuong_doc', 's5_ham', 's6_ra_ham', S7_DUNG]
