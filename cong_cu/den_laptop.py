#!/usr/bin/env python3
# CHAY_TREN: laptop
"""Nhận biển báo + đèn bằng model trên LAPTOP, đi bằng DDS (ROS 2) cho ổn định và mượt.

Thêm LiDAR và bird's eye: cũng tính + vẽ ngay trên laptop (không để xe vẽ rồi truyền qua Wi-Fi):
    /scan -> http://127.0.0.1:8096/anh/lidar.mjpg   LiDAR nhìn từ trên, vẽ cả vùng dừng cho người đi bộ của S1
    ảnh camera + calib/ -> /anh/birdeye.mjpg (bird's eye) và /anh/vach.mjpg (8 ô dò vạch như S1/S7)
Lưu ý: ô dò vạch trên laptop chạy cùng thuật toán trên cùng khung camera nhưng KHÔNG có odom/IMU của xe để dự đoán
giữa 2 khung, nên có thể lệch chút so với bộ dò thật trên xe. Dùng để XEM, xe vẫn tự lái bằng bộ dò của nó.
Luồng: /camera/image_raw/compressed (xe -> laptop, ảnh đã nén JPEG, KHÔNG giải nén lại để xem) -> model chỉ trong
vùng ROI (calib/roi_bien.yaml, đọc lại mỗi giây nên Lưu ROI có hiệu lực ngay) -> /champion/den (laptop -> xe,
JSON nhỏ trong std_msgs/String; S1 trên xe đọc). Xem hình ngay trên laptop, không qua xe:
    http://127.0.0.1:8096/anh/cam.mjpg   ảnh camera nén của xe, đúng nhịp camera
    http://127.0.0.1:8096/anh/den.mjpg   vùng ROI có vẽ khung nhận được
Model chạy ở luồng riêng, luôn lấy khung MỚI NHẤT (bỏ khung cũ) nên không bao giờ bị trễ tích luỹ.
DDS dùng ROS_DOMAIN_ID=2 và chung/fastdds_mtu.xml (gói UDP <= 1400 byte, hotspot rớt gói phân mảnh).
Bảng điều khiển tự bật script này (nút 'Nhận biển bằng laptop'); chạy tay: bash -c "source /opt/ros/humble/setup.bash; python3 cong_cu/den_laptop.py"
"""
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
os.environ.setdefault('ROS_DOMAIN_ID', '2')
os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', str(GOC / 'chung' / 'fastdds_mtu.xml'))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CompressedImage, LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402

sys.path.insert(0, str(GOC))
from chung import cau_hinh  # noqa: E402
from chung.camera import giai_nen_xam_nua  # noqa: E402
from chung.lidar import Scan  # noqa: E402
from chung.ve import lidar_tren_xuong, luoi  # noqa: E402
from chung.den_onnx import NhanDen, doc_roi, nhan_trong_roi, ve_ket_qua  # noqa: E402

CONG_XEM_LAPTOP = 8096
CHAT_LUONG_JPEG = 75


class Giu:
    """Hình mới nhất để trang xem lấy."""

    TEN = ('cam', 'den', 'lidar', 'vach', 'birdeye')

    def __init__(self):
        self.anh = {k: None for k in self.TEN}      # bản đang phát (có thể kèm banner MẤT TÍN HIỆU)
        self.goc = {k: None for k in self.TEN}      # bản JPEG mới nhất nhận được, chưa vẽ banner
        self.t = {k: 0.0 for k in self.TEN}         # giờ nhận bản gốc
        self.tt = {}

    def dat(self, ten, jpeg):
        if jpeg is None:
            return
        self.goc[ten] = jpeg
        self.anh[ten] = jpeg
        self.t[ten] = time.monotonic()


def dung_server(giu):
    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        disable_nagle_algorithm = True
        wbufsize = 1 << 16

        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/trang_thai.json':
                b = json.dumps(giu.tt).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(b)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                return self.wfile.write(b)
            if path.startswith('/anh/') and path.endswith('.mjpg'):
                ten, cu = path[5:-5], None
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.flush()
                try:
                    while True:
                        j = giu.anh.get(ten)
                        if j is not None and j is not cu:
                            self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                             + str(len(j)).encode() + b'\r\n\r\n' + j + b'\r\n')
                            self.wfile.flush()
                            cu = j
                        time.sleep(0.015)
                except OSError:
                    return
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.end_headers()

    srv = ThreadingHTTPServer(('127.0.0.1', CONG_XEM_LAPTOP), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main():
    giu = Giu()
    dung_server(giu)
    luong = max(2, (os.cpu_count() or 4) // 2)
    bo = NhanDen(('onnxruntime', 'cv2'), luong=luong)
    print(f'[den_laptop] model: {bo.ten} ({luong} luồng) | DDS domain {os.environ["ROS_DOMAIN_ID"]} | '
          f'xem: http://127.0.0.1:{CONG_XEM_LAPTOP}/anh/den.mjpg', flush=True)
    rclpy.init()
    node = rclpy.create_node('champion_den_laptop')
    pub = node.create_publisher(String, cau_hinh.TOPIC_DEN, qos_profile_sensor_data)
    moi = {'jpeg': None, 'so': 0}
    co = threading.Condition()

    def on_anh(m):
        with co:
            moi['jpeg'], moi['so'] = bytes(m.data), moi['so'] + 1
            giu.dat('cam', moi['jpeg'])           # xem ảnh nén ĐÚNG NHƯ xe gửi, không giải nén lại
            co.notify_all()

    # Driver camera phát RELIABLE: nhận RELIABLE (giữ 1 khung mới nhất) thì gói UDP rớt trên Wi-Fi được gửi lại, khung không bị
    # bỏ nguyên cái vì mất 1 trong ~25 gói (best-effort: mất 1 gói là cả khung hỏng).
    qos_cam = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(CompressedImage, cau_hinh.TOPIC_CAMERA, on_anh, qos_cam)

    def luong_model():
        roi, t_roi, so_cu, n, t_in, ms_tb = doc_roi(), 0.0, 0, 0, time.monotonic(), 0.0
        while rclpy.ok():
            with co:
                co.wait_for(lambda: moi['so'] != so_cu, timeout=1.0)
                jpeg, so = moi['jpeg'], moi['so']
            if jpeg is None or so == so_cu:
                continue
            so_cu = so
            if time.monotonic() - t_roi > 1.0:
                roi, t_roi = doc_roi(), time.monotonic()
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            kq, ms = nhan_trong_roi(bo, bgr, roi)
            pub.publish(String(data=json.dumps({'so': so, 'kq': kq, 'ms': round(ms, 1)})))
            ok, buf = cv2.imencode('.jpg', ve_ket_qua(bgr, kq, vung=roi, chi_vung=True), [cv2.IMWRITE_JPEG_QUALITY, CHAT_LUONG_JPEG])
            if ok:
                giu.dat('den', buf.tobytes())
            n += 1
            ms_tb = ms if ms_tb == 0 else 0.9 * ms_tb + 0.1 * ms
            giu.tt = {'roi': [roi[1], roi[3]], 'ms': round(ms_tb), 'khung': so}
            if time.monotonic() - t_in > 3.0:
                print(f'[den_laptop] {n / (time.monotonic() - t_in):.1f} khung/s | model {ms_tb:.0f} ms | '
                      f'ROI hàng {roi[1]}..{roi[3]}', flush=True)
                n, t_in = 0, time.monotonic()

    def ma_hoa(img):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, CHAT_LUONG_JPEG])
        return buf.tobytes() if ok else None

    # ---- LiDAR: vẽ ngay khi có scan (xe gửi /scan qua DDS) ----
    try:
        from giai_doan.s1_xuat_phat import NUA_RONG_M, nguong_dung
        non = [(0, math.degrees(math.atan2(NUA_RONG_M, nguong_dung())), nguong_dung(), (0, 255, 0))]
    except Exception:  # noqa: BLE001 - không có S1 thì vẽ không kèm vùng dừng
        non = []

    def on_scan(m):
        sc = Scan(m.angle_min + math.radians(cau_hinh.LIDAR_LECH_GOC_DO), m.angle_increment,
                  np.asarray(m.ranges, float), m.range_min, time.monotonic())
        giu.dat('lidar', ma_hoa(lidar_tren_xuong(sc, non=non, chu='LiDAR (laptop)')))

    node.create_subscription(LaserScan, cau_hinh.TOPIC_SCAN, on_scan, qos_profile_sensor_data)

    # ---- bird's eye + 8 ô dò vạch, luồng riêng, luôn lấy khung mới nhất ----
    def luong_birdeye():
        from giai_doan.s7_bam_vach import tao_bo_do
        try:
            do = tao_bo_do(False)
        except Exception as e:  # noqa: BLE001 - thiếu calib: báo rồi thôi, các hình khác vẫn chạy
            print(f'[den_laptop] KHÔNG dựng được bird\'s eye: {e}', flush=True)
            return
        so_cu = 0
        while rclpy.ok():
            with co:
                co.wait_for(lambda: moi['so'] != so_cu, timeout=1.0)
                jpeg, so = moi['jpeg'], moi['so']
            if jpeg is None or so == so_cu:
                continue
            so_cu = so
            kq = do.xu_ly(giai_nen_xam_nua(jpeg), time.monotonic(), 0.0, 0.0, True)
            giu.dat('vach', ma_hoa(luoi(list(kq.anh.items())) if kq.anh else luoi([('chua co khung debug', None)])))
            if kq.anh.get('6 fit') is not None:          # bird's eye có vạch dò được (đỏ = vạch phải)
                giu.dat('birdeye', ma_hoa(kq.anh['6 fit']))

    def canh_gac():
        """Nguồn ngừng gửi > 1 s thì VẪN phát khung cuối kèm banner MẤT TÍN HIỆU x s: hình không bao giờ biến mất/đứng im
        mà không báo, và trình duyệt luôn nhận dữ liệu (luồng không treo)."""
        while rclpy.ok():
            time.sleep(0.5)
            for ten in Giu.TEN:
                tuoi = time.monotonic() - giu.t[ten]
                g = giu.goc[ten]
                if g is None or tuoi < 1.0:
                    continue
                img = cv2.imdecode(np.frombuffer(g, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 160), -1)
                cv2.putText(img, f'MAT TIN HIEU {tuoi:.0f} s (hinh cuoi)', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                b = ma_hoa(img)
                if b is not None:
                    giu.anh[ten] = b

    threading.Thread(target=canh_gac, daemon=True).start()
    threading.Thread(target=luong_birdeye, daemon=True).start()
    threading.Thread(target=luong_model, daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
