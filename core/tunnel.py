"""Hầm bằng LiDAR: nhận cổng hầm và luật lái bám tâm hành lang (logic bản 04aba63e, chạy trên xe).

  - Nhận hầm: tường ở 2 bên (nón ±SIDE_DEG quanh ±90 độ), mỗi bên < WALL_MAX,
    tổng 2 bên = WIDTH ± WIDTH_TOL. Vào: ENTER_FRAMES khung liền; ra: mất EXIT_FRAMES khung.
  - Trong hầm: bám tâm hành lang (offset, psi, kappa đo từ tường, khớp đường tròn
    khi tường cong): kappa_cmd = K_FF*kappa + boost*K_E*offset + K_PSI*psi.
  - Khúc cua: tường phía trước < BEND_FRONT -> chậm lại, quẹo BEND_OMEGA về phía nhìn xa hơn (core/lidar.TunnelBend).
"""
from types import SimpleNamespace

from .lidar import tunnel_gate

# ===== THAM SỐ (giá trị đúng như lúc 04a chạy trên xe, car_onboard.sh) ======
WIDTH = 0.77               # tổng khoảng cách trái + phải tới tường (m)
WIDTH_TOL = 0.08
WALL_MAX = 0.70      # mỗi bên không xa hơn
SIDE_DEG = 12.0      # nửa góc nón nhìn ngang
WALL_FIT_MIN = 0.08  # điểm tường dùng để khớp: khoảng cách trong [MIN, MAX]
WALL_FIT_MAX = 0.95
WALL_FIT_MIN_PTS = 10
PSI_MAX = 0.45     # rad
KAPPA_MAX = 3.2    # 1/m
BEND_FRONT = 0.60  # tường trước < số này = khúc cua
BEND_OMEGA = 0.6   # rad/s khi quẹo ở khúc cua

V_TUNNEL = 0.13  # m/s, đúng tốc độ bản 04a
K_E = 5.0        # luật lái 04a
K_PSI = 3.5
K_FF = 1.0
EDGE_MARGIN = 0.12
EDGE_BOOST = 2.0
EDGE_BOOST_MAX = 4.0
ENTER_FRAMES = 3
EXIT_FRAMES = 6
# ===========================================================================

P = SimpleNamespace(WIDTH=WIDTH, WIDTH_TOL=WIDTH_TOL, WALL_MAX=WALL_MAX, SIDE_DEG=SIDE_DEG,
                    WALL_FIT_MIN=WALL_FIT_MIN, WALL_FIT_MAX=WALL_FIT_MAX,
                    WALL_FIT_MIN_PTS=WALL_FIT_MIN_PTS, PSI_MAX=PSI_MAX, KAPPA_MAX=KAPPA_MAX,
                    BEND_FRONT=BEND_FRONT, BEND_OMEGA=BEND_OMEGA)


class TunnelGate:
    """Đếm số khung LiDAR liên tiếp thấy cổng hầm (chỉ đếm khi có scan MỚI)."""

    def __init__(self):
        self.t_scan = None
        self.streak = 0

    def update(self, scan):
        if scan is None or scan.t == self.t_scan:
            return self.streak
        self.t_scan = scan.t
        self.streak = self.streak + 1 if tunnel_gate(scan, P) is not None else 0
        return self.streak


def steer_cmd(offset, psi, kappa, v):
    """Luật lái bản 04a (Controller.steer + apply ở trạng thái TUNNEL)."""
    excess = abs(offset) - EDGE_MARGIN
    boost = 1.0
    if excess > 0.0:
        boost = min(1.0 + EDGE_BOOST * (excess / max(EDGE_MARGIN, 1e-3)), EDGE_BOOST_MAX)
    kappa_cmd = K_FF * kappa + (boost * K_E * offset + K_PSI * psi)
    return v * kappa_cmd
