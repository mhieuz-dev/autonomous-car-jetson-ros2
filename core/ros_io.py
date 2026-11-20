"""Nối với xe thật qua ROS 2: đọc /scan /imu /odom /camera, gửi /cmd_vel.

Một node duy nhất, quay (spin) trong luồng riêng. IMU được đưa thẳng vào đối
tượng IMU ngay trong callback để không mất mẫu nào khi tích phân yaw.
Mọi xe (thật / giả lập) có cùng giao diện:
    gio(), ngu(dt), attach_imu(imu), scan, odom (Odom), camera_jpeg(), tuoi(ten),
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
class Odom:
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    w: float = 0.0
    distance: float = 0.0      # cộng dồn |ds| từ lúc mở


class RealCar:
    sim = False

    def __init__(self, need_camera=False, node_name='champion'):
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
        self.node = rclpy.create_node(node_name)
        self.imu = None
        self.scan = None
        self.odom = Odom()
        self.remote_detections = None          # kết quả nhận biển/đèn do LAPTOP gửi qua DDS: {'t', 'count', 'result', 'ms'}
        self._image = None
        self._frame_count = 0
        self._stamps = {}                 # tên cảm biến -> giờ nhận gần nhất
        self._xy_prev = None
        q = qos_profile_sensor_data
        self.node.create_subscription(Imu, settings.TOPIC_IMU, self._on_imu, q)
        self.node.create_subscription(LaserScan, settings.TOPIC_SCAN, self._on_scan, q)
        self.node.create_subscription(Odometry, settings.TOPIC_ODOM, self._on_odom, q)
        if need_camera:
            self.node.create_subscription(CompressedImage, settings.TOPIC_CAMERA, self._on_image, q)
        from std_msgs.msg import String
        self.node.create_subscription(String, settings.TOPIC_DETECTIONS, self._on_detections, q)
        self.pub = self.node.create_publisher(Twist, settings.TOPIC_CMD, 1)
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._running and self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.05)

    # ------------------------------------------------------------ callback ---
    def _on_imu(self, m):
        self._stamps['imu'] = time.monotonic()
        if self.imu is not None:
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if stamp <= 0.0:                      # driver không đóng dấu giờ -> dùng giờ nhận
                stamp = time.monotonic()
            a, g = m.linear_acceleration, m.angular_velocity
            self.imu.update(stamp, a.x, a.y, a.z, g.x, g.y, g.z)

    def _on_scan(self, m):
        self._stamps['scan'] = time.monotonic()
        self.scan = Scan(m.angle_min + math.radians(settings.LIDAR_YAW_OFFSET_DEG),
                         m.angle_increment, np.asarray(m.ranges, float),
                         m.range_min, time.monotonic())

    def _on_odom(self, m):
        self._stamps['odom'] = time.monotonic()
        p = m.pose.pose.position
        o = self.odom
        if self._xy_prev is not None:
            o.distance += math.hypot(p.x - self._xy_prev[0], p.y - self._xy_prev[1])
        self._xy_prev = (p.x, p.y)
        o.t, o.x, o.y = time.monotonic(), p.x, p.y
        o.v, o.w = m.twist.twist.linear.x, m.twist.twist.angular.z

    def _on_detections(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        self.remote_detections = {'t': time.monotonic(), 'count': d.get('count'), 'result': d.get('result', []), 'ms': d.get('ms', 0.0)}

    def _on_image(self, m):
        self._stamps['camera'] = time.monotonic()
        self._image = bytes(m.data)
        self._frame_count += 1

    # ---------------------------------------------------------------- API ---
    def now(self):
        return time.monotonic()

    def sleep(self, dt):
        time.sleep(max(0.0, dt))

    def attach_imu(self, imu):
        self.imu = imu

    def camera_jpeg(self):
        """(jpeg bytes, số thứ tự khung) của ảnh mới nhất, hoặc (None, 0)."""
        return self._image, self._frame_count

    def age(self, name):
        """Bao nhiêu giây từ lần cuối nhận cảm biến `name`; inf nếu chưa nhận lần nào."""
        t = self._stamps.get(name)
        return math.inf if t is None else time.monotonic() - t

    def send(self, v, w):
        msg = self._Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)

    def stop(self):
        for _ in range(3):
            try:
                self.send(0.0, 0.0)
            except Exception as e:      # noqa: BLE001 - dừng xe không được phép làm sập chương trình
                print(f'CẢNH BÁO: gửi lệnh dừng lỗi: {e}', flush=True)
            time.sleep(0.02)

    def close(self):
        try:
            self.stop()
        finally:
            self._running = False
            self._thread.join(timeout=1.0)
            self.node.destroy_node()
            self._rclpy.shutdown()
