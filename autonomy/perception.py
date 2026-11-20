"""Nhận biết tình huống: gom camera, LiDAR, IMU, odom thành một bức tranh (TinhHuong) cho mỗi nhịp lái.

Không giả định thứ tự sa bàn. Dốc nhận bằng pitch (đã trừ phần xe ngóc mũi theo tốc độ), hầm bằng 2 tường
LiDAR, người đi bộ bằng LiDAR phía trước, đèn và biển bằng YOLO, vạch bằng bird's eye. Mỗi phần tự theo dõi
trạng thái của nó; hành vi (autonomy/behaviors.py) chỉ đọc kết quả.
"""
import math
import threading
import time

import numpy as np

from core import tunnel
from core.lane_follow import BamVach
from core.detector import mau_den_lon
from core.lidar import CuaHam, diem_xy, tunnel_pose
from core.log import log

from . import params as TS


def nguong_nguoi():
    """Khoảng cách (tính từ LiDAR) mà người đi bộ gần hơn thì phải dừng."""
    return TS.LIDAR_TOI_CAN_M + TS.KHOANG_HO_M + TS.V_BAM_VACH * TS.THOI_GIAN_PHAN_UNG


def co_nguoi(scan):
    """(có người trước mũi?, khoảng cách gần nhất trong dải trước mũi, tính từ LiDAR)."""
    p = diem_xy(scan, 2.0) if scan is not None else []
    if len(p) == 0:
        return False, math.inf
    vung = p[(p[:, 0] > 0.0) & (np.abs(p[:, 1]) < TS.NUA_RONG_M)]
    gan = float(vung[:, 0].min()) if len(vung) else math.inf
    return int((vung[:, 0] < nguong_nguoi()).sum()) >= TS.SO_DIEM_MIN, gan


class TinhHuong:
    """Bức tranh ở một nhịp lái. NhanBiet cập nhật, hành vi chỉ đọc."""

    def __init__(self):
        self.t = 0.0
        self.imu = None
        self.v = 0.0                 # m/s theo odom
        self.quang_duong = 0.0       # m cộng dồn từ lúc mở xe
        self.scan = None
        self.chan_mui = False        # khung vừa cấm chạy tới vì vật sát mũi (core/runtime.py)
        self.pitch_do = 0.0          # pitch đã trừ phần ngóc mũi theo tốc độ (độ)
        self.doc = 'phang'           # 'phang' | 'len' | 'dinh' | 'xuong'
        self.yaw_doc = 0.0           # hướng xe lúc bắt đầu vào dốc
        self.cong_ham = 0            # số khung LiDAR liền thấy cổng hầm
        self.trong_ham = False
        self.ham_pose = None         # (offset, psi, kappa) của hành lang, đo từ tường
        self.ham_cua = None          # (v, w) khi tường chắn trước = khúc cua
        self.t_ra_ham = None         # giờ vừa ra hầm lần gần nhất
        self.yaw_ra_ham = 0.0
        self.nguoi = False
        self.nguoi_cach = math.inf
        self.bien = []               # khung YOLO mới nhất [{'ten', 'tin', 'hop'}]
        self.den = None              # 'do' | 'vang' | 'xanh' | None: đèn ĐỦ GẦN
        self.den_cao = 0.0           # px, đèn lớn nhất thấy được
        self.vach = None             # core.lane.KetQuaVach
        self.vach_moi = False        # nhịp này có khung camera mới
        self.vach_lien = 0           # số khung liền đo được vạch thật


class TheoDoiDoc:
    """phang -> len -> dinh -> xuong -> phang theo pitch đã bù tốc độ. Mỗi lần đổi, điều kiện phải đúng liên tục
    đủ lâu (DOC_*_GIAY) để rung của xe không làm đổi nhầm. Gặp dốc xuống mà không lên trước cũng nhận."""

    def __init__(self):
        self.trang_thai = 'phang'
        self.yaw_doc = 0.0
        self._ung_vien = None
        self._t_giu = 0.0
        self._yaw_bd = 0.0
        self._qd_dinh = 0.0

    def _dieu_kien(self, p):
        """[(trạng thái kế, điều kiện đúng chưa, phải giữ bao lâu)] của trạng thái hiện tại."""
        tt = self.trang_thai
        if tt == 'phang':
            return [('len', p > TS.DOC_LEN_DO, TS.DOC_LEN_GIAY), ('xuong', p < TS.DOC_XUONG_DO, TS.DOC_XUONG_GIAY)]
        if tt == 'len':
            return [('dinh', p < TS.DOC_DINH_DO, TS.DOC_DINH_GIAY)]
        if tt == 'dinh':
            return [('xuong', p < TS.DOC_XUONG_DO, TS.DOC_XUONG_GIAY)]
        return [('phang', p > TS.DOC_HET_DO, TS.DOC_HET_GIAY)]

    def cap_nhat(self, pitch_do, yaw, quang_duong, dt):
        """Trả (cũ, mới) khi vừa đổi trạng thái, không thì None."""
        if quang_duong < TS.DOC_QUANG_DUONG_MIN:
            return None
        cu = self.trang_thai
        if cu == 'dinh' and quang_duong - self._qd_dinh > TS.DINH_DAI_TOI_DA_M:
            log(f'DỐC: đỉnh dài quá {TS.DINH_DAI_TOI_DA_M} m mà chưa xuống, coi như sàn phẳng')
            return self._doi('phang')
        dung = [(ke, giu) for ke, ok, giu in self._dieu_kien(pitch_do) if ok]
        if not dung:
            self._ung_vien = None
            return None
        ke, giu = dung[0]
        if ke != self._ung_vien:
            self._ung_vien, self._t_giu, self._yaw_bd = ke, 0.0, yaw
        self._t_giu += dt
        if self._t_giu < giu:
            return None
        if cu == 'phang':
            self.yaw_doc = self._yaw_bd      # hướng lúc chân vừa chạm dốc, giữ suốt con dốc
        if ke == 'dinh':
            self._qd_dinh = quang_duong
        return self._doi(ke)

    def _doi(self, moi):
        cu, self.trang_thai, self._ung_vien = self.trang_thai, moi, None
        return cu, moi


class TheoDoiHam:
    """Vào hầm khi thấy cổng hầm ham.SO_KHUNG_VAO khung liền; ra khi mất tường ham.SO_KHUNG_RA khung
    (tường chắn phía trước ở khúc cua không tính là mất)."""

    def __init__(self):
        self.dem = tunnel.DemHam()
        self.cua = CuaHam()
        self.trong = False
        self.mat = 0
        self.pose = None
        self.cua_lenh = None
        self._t_scan = None

    def cap_nhat(self, scan):
        """Trả 'vao' / 'ra' khi vừa vào / ra hầm, không thì None."""
        cong = self.dem.cap_nhat(scan)
        if scan is None or scan.t == self._t_scan:
            return None
        self._t_scan = scan.t
        if not self.trong:
            if cong < tunnel.SO_KHUNG_VAO:
                return None
            self.trong, self.mat, self.pose, self.cua = True, 0, None, CuaHam()
            return 'vao'
        pose = tunnel_pose(scan, tunnel.P)
        self.cua_lenh = self.cua.tinh(scan, tunnel.P, tunnel.V_HAM)
        if pose is not None:
            self.pose, self.mat = pose, 0
        elif self.cua_lenh is not None:
            self.mat = 0
        else:
            self.mat += 1
        if self.mat < tunnel.SO_KHUNG_RA:
            return None
        self.trong, self.pose, self.cua_lenh = False, None, None
        return 'ra'


class NhanBien:
    """Đèn + biển bằng YOLO. Trên xe chạy ở luồng riêng (TensorRT ~100 ms/khung), mỗi nhịp lái chỉ lấy kết quả
    mới nhất, cũ quá 2 s coi như không thấy. Sa bàn giả lấy thẳng đèn / biển dựng sẵn (xe.den_doc_duong)."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.chay = True
        self._kq, self._t = [], 0.0
        if ctx.gia_lap or not (TS.DEN_BAT or TS.BIEN_BAT):
            return
        if TS.DEN_NGUON == 'laptop':
            log('Nhận đèn/biển bằng model trên LAPTOP qua DDS (bảng điều khiển phải bật "Nhận biển bằng laptop")')
            return
        from core.detector import NhanDen, doc_roi
        self.roi = doc_roi()
        self.bo = NhanDen()
        log(f'Nhận đèn/biển bằng {self.bo.ten}, ROI hàng {self.roi[1]}..{self.roi[3]}'
            + (f' | lỗi đã gặp: {self.bo.loi}' if self.bo.loi else ''))
        if not self.bo.san_sang:
            log('CẢNH BÁO: không chạy được model đèn/biển, xe chạy KHÔNG nhận đèn và biển')
            return
        threading.Thread(target=self._vong, daemon=True).start()

    def ket_qua(self):
        ctx = self.ctx
        if ctx.gia_lap:
            return ctx.xe.den_doc_duong()
        if TS.DEN_NGUON == 'laptop':
            n = getattr(ctx.xe, 'den_ngoai', None)
            return n['kq'] if n is not None and time.monotonic() - n['t'] < 2.0 else []
        return self._kq if time.monotonic() - self._t < 2.0 else []

    def _vong(self):
        from core.camera import giai_nen_mau
        from core.detector import nhan_trong_roi, ve_ket_qua
        so_cu = 0
        while self.chay:
            jpeg, so = self.ctx.xe.anh_jpeg()
            if jpeg is None or so == so_cu:
                time.sleep(0.03)
                continue
            so_cu = so
            t_bd = time.monotonic()
            bgr = giai_nen_mau(jpeg)
            kq, _ = nhan_trong_roi(self.bo, bgr, self.roi)
            self._kq, self._t = kq, time.monotonic()
            if self.ctx.xem is not None:
                self.ctx.xem.dat_anh('den', ve_ket_qua(bgr, kq, vung=self.roi, chi_vung=True))
            con_lai = TS.DEN_CHU_KY - (time.monotonic() - t_bd)
            if con_lai > 0:             # model chậm hơn chu kỳ thì chạy liên tục, không ngủ thêm
                time.sleep(con_lai)


class NhanBiet:
    def __init__(self, ctx):
        self.th = TinhHuong()
        self.th.imu = ctx.imu
        self.bam = BamVach(ctx.gia_lap)
        self.doc = TheoDoiDoc()
        self.ham = TheoDoiHam()
        self.bien = NhanBien(ctx)
        self._v2 = 0.0

    def cap_nhat(self, ctx, dt):
        th, xe, imu = self.th, ctx.xe, ctx.imu
        th.t = xe.gio()
        th.v, th.quang_duong, th.scan = xe.odom.v, xe.odom.quang_duong, xe.scan
        th.chan_mui = ctx.chan_mui

        self._v2 += (th.v * th.v - self._v2) * min(1.0, dt / TS.NGOC_MUI_TRE_S)
        th.pitch_do = math.degrees(imu.pitch) - TS.NGOC_MUI_DO * self._v2
        doi = self.doc.cap_nhat(th.pitch_do, imu.yaw, th.quang_duong, dt)
        if doi:
            log(f'DỐC: {doi[0]} -> {doi[1]} (pitch {th.pitch_do:+.1f} độ đã bù tốc độ, '
                f'giữ hướng {math.degrees(self.doc.yaw_doc):+.1f} độ)')
        th.doc, th.yaw_doc = self.doc.trang_thai, self.doc.yaw_doc

        su_kien = self.ham.cap_nhat(xe.scan)
        if su_kien == 'vao':
            log('HẦM: vào hầm (LiDAR thấy 2 tường đúng bề rộng)')
        elif su_kien == 'ra':
            th.t_ra_ham, th.yaw_ra_ham = th.t, imu.yaw
            self.bam.do.dat_lai()           # trong hầm vạch đo bậy, dò lại từ đầu
            th.vach_lien = 0
            log(f'HẦM: ra hầm (mất tường {tunnel.SO_KHUNG_RA} khung), hướng {math.degrees(imu.yaw):+.1f} độ')
        th.cong_ham, th.trong_ham = self.ham.dem.lien, self.ham.trong
        th.ham_pose, th.ham_cua = self.ham.pose, self.ham.cua_lenh

        nguoi, th.nguoi_cach = co_nguoi(xe.scan)
        # tường / mặt dốc ở cổng hầm hay bị nhận là người; trên dốc thì luật cấm dừng
        th.nguoi = nguoi and th.doc == 'phang' and not th.trong_ham and th.cong_ham < 2

        th.bien = self.bien.ket_qua()
        th.den, th.den_cao, _ = mau_den_lon(th.bien, TS.DEN_CAO_MIN_PX, TS.DEN_TIN_MIN, TS.DEN_DO_TIN_MIN)

        debug = ctx.xem is not None and ctx.xem.dang_xem('vach')
        th.vach, th.vach_moi = self.bam.cap_nhat(ctx, debug=debug)
        if th.vach_moi:
            th.vach_lien = th.vach_lien + 1 if th.vach.do_moi else 0
        return th

    def dung(self):
        self.bien.chay = False
