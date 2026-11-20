#!/usr/bin/env python3
# RUNS_ON: laptop
"""Nhận biển báo + đèn bằng model trên LAPTOP, đi bằng DDS (ROS 2) cho ổn định và mượt.

Thêm LiDAR và bird's eye: cũng tính + vẽ ngay trên laptop (không để xe vẽ rồi truyền qua Wi-Fi):
    /scan -> http://127.0.0.1:8096/img/lidar.mjpg   LiDAR nhìn từ trên, vẽ cả vùng dừng cho người đi bộ
    ảnh camera + config/ -> /img/birdeye.mjpg (bird's eye) và /img/line.mjpg (8 ô dò vạch như trên xe)
Lưu ý: ô dò vạch trên laptop chạy cùng thuật toán trên cùng khung camera nhưng KHÔNG có odom/IMU của xe để dự đoán
giữa 2 khung, nên có thể lệch chút so với bộ dò thật trên xe. Dùng để XEM, xe vẫn tự lái bằng bộ dò của nó.
Luồng: /camera/image_raw/compressed (xe -> laptop, ảnh đã nén JPEG, KHÔNG giải nén lại để xem) -> model chỉ trong
vùng ROI (config/sign_roi.yaml, đọc lại mỗi giây nên Lưu ROI có hiệu lực ngay) -> /champion/den (laptop -> xe,
JSON nhỏ trong std_msgs/String; xe tự hành đọc khi LIGHT_SOURCE = "laptop"). Xem hình ngay trên laptop, không qua xe:
    http://127.0.0.1:8096/img/cam.mjpg   ảnh camera nén của xe, đúng nhịp camera
    http://127.0.0.1:8096/img/light.mjpg   vùng ROI có vẽ khung nhận được
Model chạy ở luồng riêng, luôn lấy khung MỚI NHẤT (bỏ khung cũ) nên không bao giờ bị trễ tích luỹ.
DDS dùng ROS_DOMAIN_ID=2 và config/fastdds_mtu.xml (gói UDP <= 1400 byte, hotspot rớt gói phân mảnh).
Bảng điều khiển tự bật script này (nút 'Nhận biển bằng laptop'); chạy tay: bash -c "source /opt/ros/humble/setup.bash; python3 tools/detector_laptop.py"
"""
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('ROS_DOMAIN_ID', '2')
os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', str(ROOT / 'core' / 'fastdds_mtu.xml'))

import cv2                                                                                   # noqa: E402
import numpy as np                                                                           # noqa: E402
import rclpy                                                                                 # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CompressedImage, LaserScan                                       # noqa: E402
from std_msgs.msg import String                                                              # noqa: E402

sys.path.insert(0, str(ROOT))
from core import settings                                                     # noqa: E402
from core.camera import decode_gray_half                                      # noqa: E402
from core.lidar import Scan                                                   # noqa: E402
from core.draw import lidar_top_view, grid                                    # noqa: E402
from core.detector import Detector, read_roi, detect_in_roi, draw_detections  # noqa: E402

LAPTOP_VIEW_PORT = 8096
JPEG_QUALITY = 75


class FrameStore:
    """Hình mới nhất để trang xem lấy."""

    NAME = ('cam', 'light', 'lidar', 'line', 'birdeye')

    def __init__(self):
        self.image = {k: None for k in self.NAME}   # bản đang phát (có thể kèm banner MẤT TÍN HIỆU)
        self.origin = {k: None for k in self.NAME}  # bản JPEG mới nhất nhận được, chưa vẽ banner
        self.t = {k: 0.0 for k in self.NAME}        # giờ nhận bản gốc
        self.status = {}

    def put(self, name, jpeg):
        if jpeg is None:
            return
        self.origin[name] = jpeg
        self.image[name] = jpeg
        self.t[name] = time.monotonic()


def serve(store):
    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        disable_nagle_algorithm = True
        wbufsize = 1 << 16

        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/status.json':
                b = json.dumps(store.status).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(b)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                return self.wfile.write(b)
            if path.startswith('/img/') and path.endswith('.mjpg'):
                name, old = path[5:-5], None
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.flush()
                try:
                    while True:
                        j = store.image.get(name)
                        if j is not None and j is not old:
                            self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                             + str(len(j)).encode() + b'\r\n\r\n' + j + b'\r\n')
                            self.wfile.flush()
                            old = j
                        time.sleep(0.015)
                except OSError:
                    return
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.end_headers()

    srv = ThreadingHTTPServer(('127.0.0.1', LAPTOP_VIEW_PORT), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def main():
    store = FrameStore()
    serve(store)
    thread = max(2, (os.cpu_count() or 4) // 2)
    backend = Detector(('onnxruntime', 'cv2'), thread=thread)
    print(f'[detector_laptop] model: {backend.name} ({thread} luồng) | DDS domain {os.environ["ROS_DOMAIN_ID"]} | '
          f'xem: http://127.0.0.1:{LAPTOP_VIEW_PORT}/img/light.mjpg', flush=True)
    rclpy.init()
    node = rclpy.create_node('champion_detector_laptop')
    pub = node.create_publisher(String, settings.TOPIC_DETECTIONS, qos_profile_sensor_data)
    new = {'jpeg': None, 'count': 0}
    size = threading.Condition()

    def on_image(m):
        with size:
            new['jpeg'], new['count'] = bytes(m.data), new['count'] + 1
            store.put('cam', new['jpeg'])           # xem ảnh nén ĐÚNG NHƯ xe gửi, không giải nén lại
            size.notify_all()

    # Driver camera phát RELIABLE: nhận RELIABLE (giữ 1 khung mới nhất) thì gói UDP rớt trên Wi-Fi được gửi lại, khung không bị
    # bỏ nguyên cái vì mất 1 trong ~25 gói (best-effort: mất 1 gói là cả khung hỏng).
    qos_cam = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(CompressedImage, settings.TOPIC_CAMERA, on_image, qos_cam)

    def model_thread():
        roi, t_roi, old_count, n, t_in, ms_mean = read_roi(), 0.0, 0, 0, time.monotonic(), 0.0
        while rclpy.ok():
            with size:
                size.wait_for(lambda: new['count'] != old_count, timeout=1.0)
                jpeg, count = new['jpeg'], new['count']
            if jpeg is None or count == old_count:
                continue
            old_count = count
            if time.monotonic() - t_roi > 1.0:
                roi, t_roi = read_roi(), time.monotonic()
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            result, ms = detect_in_roi(backend, bgr, roi)
            pub.publish(String(data=json.dumps({'count': count, 'result': result, 'ms': round(ms, 1)})))
            ok, buf = cv2.imencode('.jpg', draw_detections(bgr, result, zone=roi, zone_only=True), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                store.put('light', buf.tobytes())
            n += 1
            ms_mean = ms if ms_mean == 0 else 0.9 * ms_mean + 0.1 * ms
            store.status = {'roi': [roi[1], roi[3]], 'ms': round(ms_mean), 'frame': count}
            if time.monotonic() - t_in > 3.0:
                print(f'[detector_laptop] {n / (time.monotonic() - t_in):.1f} khung/s | model {ms_mean:.0f} ms | '
                      f'ROI hàng {roi[1]}..{roi[3]}', flush=True)
                n, t_in = 0, time.monotonic()

    def encode(img):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None

    # ---- LiDAR: vẽ ngay khi có scan (xe gửi /scan qua DDS) ----
    try:
        from autonomy.perception import pedestrian_threshold
        from autonomy.params import HALF_WIDTH_M
        cones = [(0, math.degrees(math.atan2(HALF_WIDTH_M, pedestrian_threshold())), pedestrian_threshold(), (0, 255, 0))]
    except Exception:  # noqa: BLE001 - không đọc được tham số thì vẽ không kèm vùng dừng
        cones = []

    def on_scan(m):
        sc = Scan(m.angle_min + math.radians(settings.LIDAR_YAW_OFFSET_DEG), m.angle_increment,
                  np.asarray(m.ranges, float), m.range_min, time.monotonic())
        store.put('lidar', encode(lidar_top_view(sc, cones=cones, text='LiDAR (laptop)')))

    node.create_subscription(LaserScan, settings.TOPIC_SCAN, on_scan, qos_profile_sensor_data)

    # ---- bird's eye + 8 ô dò vạch, luồng riêng, luôn lấy khung mới nhất ----
    def birdeye_thread():
        from core.lane_follow import make_detector
        try:
            det = make_detector(False)
        except Exception as e:  # noqa: BLE001 - thiếu calib: báo rồi thôi, các hình khác vẫn chạy
            print(f'[detector_laptop] KHÔNG dựng được bird\'s eye: {e}', flush=True)
            return
        old_count = 0
        while rclpy.ok():
            with size:
                size.wait_for(lambda: new['count'] != old_count, timeout=1.0)
                jpeg, count = new['jpeg'], new['count']
            if jpeg is None or count == old_count:
                continue
            old_count = count
            result = det.process(decode_gray_half(jpeg), time.monotonic(), 0.0, 0.0, True)
            store.put('line', encode(grid(list(result.image.items())) if result.image else grid([('chua co khung debug', None)])))
            if result.image.get('6 fit') is not None:          # bird's eye có vạch dò được (đỏ = vạch phải)
                store.put('birdeye', encode(result.image['6 fit']))

    def watchdog():
        """Nguồn ngừng gửi > 1 s thì VẪN phát khung cuối kèm banner MẤT TÍN HIỆU x s: hình không bao giờ biến mất/đứng im
        mà không báo, và trình duyệt luôn nhận dữ liệu (luồng không treo)."""
        while rclpy.ok():
            time.sleep(0.5)
            for name in FrameStore.NAME:
                age = time.monotonic() - store.t[name]
                g = store.origin[name]
                if g is None or age < 1.0:
                    continue
                img = cv2.imdecode(np.frombuffer(g, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 160), -1)
                cv2.putText(img, f'MAT TIN HIEU {age:.0f} s (hinh cuoi)', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                b = encode(img)
                if b is not None:
                    store.image[name] = b

    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=birdeye_thread, daemon=True).start()
    threading.Thread(target=model_thread, daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
