"""Cấu hình chung cho mọi script: đường dẫn, chủ đề ROS, cổng xem hình."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Chủ đề ROS trên xe thật (đo 09/10: camera 30 Hz, /scan 10 Hz, /imu và /odom 25 Hz)
TOPIC_CAMERA = '/camera/image_raw/compressed'
TOPIC_SCAN = '/scan'
TOPIC_IMU = '/imu'
TOPIC_ODOM = '/odom'
TOPIC_CMD = '/cmd_vel'
TOPIC_DETECTIONS = '/champion/den'            # LAPTOP -> xe: kết quả nhận biển/đèn (JSON trong std_msgs/String), xem tools/detector_laptop.py

# Driver BTC đã xoay /scan 180 độ (CRC_SCAN_YAW_OFFSET_DEG=180 trong ~/crc_car.env).
# Nếu lidar_test cho thấy "trước" và "sau" bị ngược thì đặt 180 ở đây.
LIDAR_YAW_OFFSET_DEG = 0.0

# Kích thước xe tính từ tâm LiDAR. ĐO LẠI BẰNG THƯỚC (số đang là tượng trưng)!
LIDAR_TO_BUMPER_M = 0.197       # tâm LiDAR -> mép cản TRƯỚC
LIDAR_TO_REAR_BUMPER_M = 0.069  # tâm LiDAR -> mép cản SAU

# CHẶN SÁT MŨI (mọi script lái, không điều kiện): vật trong nón ±NOSE_GUARD_HALF_DEG
# cách cản trước < NOSE_GUARD_DIST_M thì CẤM CHẠY TỚI (v = 0). Vẫn được quay tại chỗ
# và lùi, nên khúc cua hầm vẫn xoay ra được. Vật đi khỏi thì chạy tiếp.
NOSE_GUARD_ON = False   # tắt mặc định: LiDAR chạy đúng bản 04a. Bật = True nếu muốn thử
NOSE_GUARD_DIST_M = 0.05
NOSE_GUARD_HALF_DEG = 20.0

VIEW_PORT = 8097                        # trang hình + trạng thái của script đang chạy
STOP_FILE = Path('/tmp/champion_stop')  # có file này là mọi script dừng xe ngay

LOG_DIR = ROOT / 'logs'
CONFIG_DIR = ROOT / 'config'        # calib ĐANG DÙNG (commit lên git)
CALIB_NEW_DIR = ROOT / 'calib_new'  # calib vừa đo trên xe, chờ duyệt (`scripts/car.sh accept_calib`)
MODEL_ONNX = ROOT / 'models' / 'best.onnx'
MODEL_ENGINE = ROOT / 'models' / 'best.engine'

# Vòng gyro (chung cho mọi chương trình lái): bù cho bánh lệch / trượt bằng gyro z.
# Bản cũ chạy cái này trên xe (imu_yaw_loop.py, kp 0.5). Tắt = GYRO_LOOP_ON False.
GYRO_LOOP_ON = True
GYRO_KP = 0.5
GYRO_KI = 1.0
GYRO_I_MAX = 0.3
