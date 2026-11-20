"""Nối với xe thật qua ROS 2: đọc /scan /imu /odom /camera, gửi /cmd_vel.

Một node duy nhất, quay (spin) trong luồng riêng. IMU được đưa thẳng vào đối
tượng IMU ngay trong callback để không mất mẫu nào khi tích phân yaw.
Mọi xe (thật / giả lập) có cùng giao diện:
    gio(), ngu(dt), gan_imu(imu), scan, odom (OdomTin), anh_jpeg(), tuoi(ten),
    gui(v, w), dung(), dong()
"""
import json
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import settings
from .lidar import Scan


@dataclass
class OdomTin:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    w: float = 0.0
    quang_duong: float = 0.0      # cộng dồn |ds| từ lúc mở


class XeThat:
    gia_lap = False

    def __init__(self, can_camera=False, ten_node='champion'):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Imu, LaserScan

        self._rclpy = rclpy
        self._Twist = Twist
        # ROS mặc định tự tắt kết nối khi nhận Ctrl+C, nên lệnh DỪNG XE không gửi đi được (RCLError). Tắt cơ chế đó:
        # tín hiệu do runtime.py xử lý, vòng lặp thoát có trật tự rồi mới gửi v = w = 0 và đóng.
        try:
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        except ImportError:
            rclpy.init()
        self.node = rclpy.create_node(ten_node)
        self.imu = None
        self.scan = None
        self.odom = OdomTin()
        self.den_ngoai = None          # kết quả nhận biển/đèn do LAPTOP gửi qua DDS: {'t', 'so', 'kq', 'ms'}
        self._anh = None
        self._so_anh = 0
        self._luc = {}                 # tên cảm biến -> giờ nhận gần nhất
        self._xy_truoc = None
        q = qos_profile_sensor_data
        self.node.create_subscription(Imu, settings.TOPIC_IMU, self._on_imu, q)
        self.node.create_subscription(LaserScan, settings.TOPIC_SCAN, self._on_scan, q)
        self.node.create_subscription(Odometry, settings.TOPIC_ODOM, self._on_odom, q)
        if can_camera:
            self.node.create_subscription(CompressedImage, settings.TOPIC_CAMERA, self._on_anh, q)
        from std_msgs.msg import String
        self.node.create_subscription(String, settings.TOPIC_DEN, self._on_den, q)
        self.pub = self.node.create_publisher(Twist, settings.TOPIC_LENH, 1)
        self._chay = True
        self._luong = threading.Thread(target=self._spin, daemon=True)
        self._luong.start()

    def _spin(self):
        while self._chay and self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.05)

    # ------------------------------------------------------------ callback ---
    def _on_imu(self, m):
        self._luc['imu'] = time.monotonic()
        if self.imu is not None:
            st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if st <= 0.0:                      # driver không đóng dấu giờ -> dùng giờ nhận
                st = time.monotonic()
            a, g = m.linear_acceleration, m.angular_velocity
            self.imu.cap_nhat(st, a.x, a.y, a.z, g.x, g.y, g.z)

    def _on_scan(self, m):
        self._luc['scan'] = time.monotonic()
        self.scan = Scan(m.angle_min + math.radians(settings.LIDAR_LECH_GOC_DO),
                         m.angle_increment, np.asarray(m.ranges, float),
                         m.range_min, time.monotonic())

    def _on_odom(self, m):
        self._luc['odom'] = time.monotonic()
        p = m.pose.pose.position
        o = self.odom
        if self._xy_truoc is not None:
            o.quang_duong += math.hypot(p.x - self._xy_truoc[0], p.y - self._xy_truoc[1])
        self._xy_truoc = (p.x, p.y)
        o.t, o.x, o.y = time.monotonic(), p.x, p.y
        o.v, o.w = m.twist.twist.linear.x, m.twist.twist.angular.z

    def _on_den(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        self.den_ngoai = {'t': time.monotonic(), 'so': d.get('so'), 'kq': d.get('kq', []), 'ms': d.get('ms', 0.0)}

    def _on_anh(self, m):
        self._luc['camera'] = time.monotonic()
        self._anh = bytes(m.data)
        self._so_anh += 1

    # ---------------------------------------------------------------- API ---
    def gio(self):
        return time.monotonic()

    def ngu(self, dt):
        time.sleep(max(0.0, dt))

    def gan_imu(self, imu):
        self.imu = imu

    def anh_jpeg(self):
        """(jpeg bytes, số thứ tự khung) của ảnh mới nhất, hoặc (None, 0)."""
        return self._anh, self._so_anh

    def tuoi(self, ten):
        """Bao nhiêu giây từ lần cuối nhận cảm biến `ten`; inf nếu chưa nhận lần nào."""
        t = self._luc.get(ten)
        return math.inf if t is None else time.monotonic() - t

    def gui(self, v, w):
        msg = self._Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)

    def dung(self):
        for _ in range(3):
            try:
                self.gui(0.0, 0.0)
            except Exception as e:      # noqa: BLE001 - dừng xe không được phép làm sập chương trình
                print(f'CẢNH BÁO: gửi lệnh dừng lỗi: {e}', flush=True)
            time.sleep(0.02)

    def dong(self):
        try:
            self.dung()
        finally:
            self._chay = False
            self._luong.join(timeout=1.0)
            self.node.destroy_node()
            self._rclpy.shutdown()
