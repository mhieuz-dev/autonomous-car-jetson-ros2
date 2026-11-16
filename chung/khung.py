"""Khung chạy giai đoạn: mở xe, hiệu chuẩn IMU, chạy lần lượt từng giai đoạn.

Mỗi giai đoạn (giai_doan/sX_*.py) là một lớp GiaiDoan với:
    TEN, HZ, THOI_GIAN_TOI_DA, CAN_CAMERA
    bat_dau(ctx)          gọi 1 lần lúc vào giai đoạn
    buoc(ctx, dt)         trả (v, w), (v, w, 'gap') = đặt ngay không qua giới hạn
                          gia tốc (dừng gấp), hoặc KetThuc(...)
    trang_thai()          chuỗi ngắn đưa vào log
    ve(ctx)               (tuỳ chọn) đặt hình lên trang xem
Khung lo: kiểm cảm biến, dừng khẩn (Ctrl+C / file /tmp/champion_dung / mất
cảm biến), giới hạn gia tốc, log mỗi 0,5 s, ghi bag, gửi trạng thái lên trang xem.
Khi qua giai đoạn mới, đối tượng giai đoạn cũ bị bỏ: logic cũ KHÔNG chạy nữa.
"""
import argparse
import importlib
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import cau_hinh
from .imu import IMU
from .lidar import range_in_cone
from .log import log, mo_file
from .pid import GioiHan, VongGyro


class KetThuc:
    def __init__(self, ly_do, tiep=None, dung_han=False):
        self.ly_do = ly_do
        self.tiep = tiep            # tên giai đoạn nhảy tới (None = giai đoạn kế)
        self.dung_han = dung_han    # True = dừng cả lượt chạy


class GiaiDoanCoSo:
    TEN = '?'
    HZ = 20
    THOI_GIAN_TOI_DA = 60.0
    CAN_CAMERA = False
    GIOI_HAN = dict(gia_toc=0.6, w_max=1.5, w_doi=6.0)

    def bat_dau(self, ctx):
        pass

    def buoc(self, ctx, dt):
        raise NotImplementedError

    def trang_thai(self):
        return ''

    def ve(self, ctx):
        pass

    def ket_thuc(self, ctx, ly_do):
        """Gọi khi giai đoạn kết thúc vì BẤT KỲ lý do nào (xong, dừng tay, hết giờ, mất cảm biến)."""
        pass


class NguCanh:
    def __init__(self, xe, imu, xem, gia_lap):
        self.xe, self.imu, self.xem, self.gia_lap = xe, imu, xem, gia_lap
        self.du_lieu = {}           # giai đoạn trước để lại cho giai đoạn sau
        self.v_cuoi = self.w_cuoi = 0.0
        self.chan_mui = self._chan_mui_cu = False
        self._qd0 = 0.0
        self.t0 = 0.0
        self.nhan_nen = None        # luồng nhận biển nền (STOP ở S5-S7), bật khi cần
        self.dung_stop = None

    def quang_duong(self):
        """Mét đã chạy từ đầu giai đoạn hiện tại (odom)."""
        return self.xe.odom.quang_duong - self._qd0

    def thoi_gian(self):
        return self.xe.gio() - self.t0


_DUNG = {'co': False, 'ly_do': ''}


def _bat_tin_hieu():
    def h(sig, _frame):
        _DUNG['co'] = True
        _DUNG['ly_do'] = signal.Signals(sig).name
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, h)


def can_dung():
    if _DUNG['co']:
        return f'dừng tay ({_DUNG["ly_do"]})'
    if cau_hinh.FILE_DUNG.exists():
        return 'dừng khẩn (file /tmp/champion_dung)'
    return None


def cho_cam_bien(xe, ten, toi_da=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < toi_da:
        thieu = [n for n in ten if xe.tuoi(n) > 1.0]
        if not thieu:
            return []
        if can_dung():
            return thieu
        xe.ngu(0.05)
    return [n for n in ten if xe.tuoi(n) > 1.0]


def hieu_chuan_imu(xe, imu, giay=2.0):
    """Xe đứng yên `giay` giây: đo bias gyro, đặt yaw = 0 theo mũi xe, pitch = 0."""
    log(f'IMU: đứng yên {giay:.0f} s để hiệu chuẩn (đừng chạm xe)...')
    imu.bat_dau_gom()
    t0 = xe.gio()
    while xe.gio() - t0 < giay:
        xe.gui(0.0, 0.0)
        xe.ngu(0.05)
    so = imu.ket_thuc_gom()
    log(f'IMU: {so["so_mau"]} mẫu | trọng lực {so["trong_luc"]:.2f} m/s2 | '
        f'bias gz {so["bias_gz"]:+.4f} rad/s | rung gz {so["rung_gz"]:.4f} | yaw = 0, pitch = 0')
    return so


def _ghi_bag(ten):
    cau_hinh.THU_MUC_LOG.mkdir(parents=True, exist_ok=True)
    out = cau_hinh.THU_MUC_LOG / f'bag_{time.strftime("%Y%m%d_%H%M%S")}_{ten}'
    topics = [cau_hinh.TOPIC_SCAN, cau_hinh.TOPIC_IMU, cau_hinh.TOPIC_ODOM,
              cau_hinh.TOPIC_LENH, cau_hinh.TOPIC_CAMERA]
    try:
        p = subprocess.Popen(['ros2', 'bag', 'record', '-o', str(out), *topics],
                             stdout=open(str(out) + '.log', 'w'), stderr=subprocess.STDOUT)
        log(f'Ghi bag: {out}')
        return p
    except OSError as e:
        log(f'CẢNH BÁO: không ghi được bag ({e})')
        return None


def _stop_ca_map(gd, ctx):
    """Biển STOP cho các giai đoạn trong S1.STOP_CA_MAP (S1 tự xử lý STOP của nó, dốc S2-S4 không dừng)."""
    from giai_doan import s1_xuat_phat as S1
    if not S1.BIEN_BAT or gd.TEN not in S1.STOP_CA_MAP:
        return False
    if ctx.nhan_nen is None:
        from .bien_stop import DungStop, NhanNen
        ctx.nhan_nen = NhanNen(ctx)
        ctx.dung_stop = ctx.dung_stop or DungStop()
    return ctx.dung_stop.buoc(ctx.nhan_nen.kq(), ctx.xe.gio(), gd.TEN, S1.STOP_CAO_MIN_PX, S1.BIEN_TIN_MIN,
                              S1.STOP_DUNG_GIAY, S1.STOP_BO_QUA_GIAY)


def _chay_mot(gd, ctx):
    xe = ctx.xe
    gh = GioiHan(**gd.GIOI_HAN)
    gh.dat_lai(ctx.v_cuoi, ctx.w_cuoi)
    gyro = VongGyro(cau_hinh.GYRO_KP, cau_hinh.GYRO_KI, cau_hinh.GYRO_I_MAX) if cau_hinh.BAT_VONG_GYRO else None
    ctx._qd0 = xe.odom.quang_duong
    ctx.t0 = xe.gio()
    gd.bat_dau(ctx)
    can = ['imu', 'scan', 'odom'] + (['camera'] if gd.CAN_CAMERA else [])
    chu_ky = 1.0 / gd.HZ
    t_truoc = xe.gio()
    t_log = t_ve = -1e9
    while True:
        ly_do = can_dung()
        if ly_do:
            return KetThuc(ly_do, dung_han=True)
        cu = [n for n in can if xe.tuoi(n) > (1.0 if n == 'camera' else 0.5)]
        if cu:
            xe.dung()
            return KetThuc(f'MẤT CẢM BIẾN {cu}', dung_han=True)
        bay_gio = xe.gio()
        dt = max(1e-3, bay_gio - t_truoc)
        t_truoc = bay_gio
        if bay_gio - ctx.t0 > gd.THOI_GIAN_TOI_DA:
            xe.dung()
            return KetThuc(f'quá thời gian tối đa {gd.THOI_GIAN_TOI_DA:.0f} s', dung_han=True)
        kq = gd.buoc(ctx, dt)
        if isinstance(kq, KetThuc):
            ctx.v_cuoi, ctx.w_cuoi = gh.v, gh.w
            return kq
        if _stop_ca_map(gd, ctx):
            kq = (0.0, 0.0, 'gap')               # biển STOP đủ gần: đứng yên (giai đoạn vẫn chạy bên trong)
        ctx.chan_mui = False
        if cau_hinh.CHAN_MUI_BAT and kq[0] > 0.0:
            d = range_in_cone(xe.scan, 0.0, cau_hinh.CHAN_MUI_NUA_GOC_DO)
            if d is not None and d < cau_hinh.LIDAR_TOI_CAN_M + cau_hinh.CHAN_MUI_CACH_M:
                ctx.chan_mui = True
                kq = (0.0, kq[1], 'gap')         # cấm chạy tới, vẫn cho quay
        if ctx.chan_mui != ctx._chan_mui_cu:
            ctx._chan_mui_cu = ctx.chan_mui
            log('CHẶN MŨI: vật sát cản trước -> không chạy tới (vẫn quay / lùi được)' if ctx.chan_mui
                else 'CHẶN MŨI: hết vật cản -> chạy tiếp')
        if len(kq) == 3 and kq[2] == 'gap':      # dừng gấp: bỏ qua giới hạn gia tốc
            gh.dat_lai(kq[0], kq[1])
            v, w = kq[0], kq[1]
        else:
            v, w = gh.ap(kq[0], kq[1], dt)
        w_gui = gyro.ap(v, w, ctx.imu.gz, dt) if gyro else w
        if not getattr(gd, 'KHONG_GUI', False):       # chế độ ĐẨY TAY: không gửi lệnh nào để bánh quay tự do
            xe.gui(v, w_gui)
        imu = ctx.imu
        tt = gd.trang_thai()
        if ctx.xem is not None:
            ctx.xem.cap_nhat(giai_doan=gd.TEN, v=round(v, 3), w=round(w, 3),
                             yaw=round(math.degrees(imu.yaw), 1), pitch=round(math.degrees(imu.pitch), 2),
                             roll=round(math.degrees(imu.roll), 2), quang_duong=round(ctx.quang_duong(), 2),
                             t=round(ctx.thoi_gian(), 1), chi_tiet=tt)
            if bay_gio - t_ve >= 0.2:
                t_ve = bay_gio
                gd.ve(ctx)
        if bay_gio - t_log >= 0.5:
            t_log = bay_gio
            log(f'{gd.TEN:13s} t={ctx.thoi_gian():5.1f} v={v:+.2f} w={w:+.2f} '
                f'yaw={math.degrees(imu.yaw):+6.1f} pitch={math.degrees(imu.pitch):+5.1f} '
                f'qd={ctx.quang_duong():4.2f}m | {tt}')
        con_lai = chu_ky - (xe.gio() - bay_gio) if not xe.gia_lap else chu_ky
        xe.ngu(max(0.0, con_lai))


def chay_mot(gd, ctx):
    kq = _chay_mot(gd, ctx)
    try:
        gd.ket_thuc(ctx, kq.ly_do)
    except Exception as e:      # noqa: BLE001 - báo cáo lỗi không được làm hỏng việc dừng xe
        log(f'CẢNH BÁO: lỗi trong ket_thuc của {gd.TEN}: {e}')
    return kq


def _lop(mod):
    """Lớp chạy được trong module: ChuongTrinh (tu_hanh/lai_xe.py) hoặc GiaiDoan (giai_doan/, cong_cu/imu_doc.py)."""
    return getattr(mod, 'ChuongTrinh', None) or mod.GiaiDoan


def chay(ds_ten, gia_lap=False, ghi_bag=True, nhanh=False, xem_cong=cau_hinh.CONG_XEM, xe=None):
    """Chạy lần lượt các giai đoạn trong ds_ten. Trả (mã thoát, ngữ cảnh)."""
    _bat_tin_hieu()
    _DUNG['co'] = False
    try:
        cau_hinh.FILE_DUNG.unlink()
    except OSError:
        pass
    ten_log = ds_ten[0] if len(ds_ten) == 1 else f'tu_{ds_ten[0]}'
    if not nhanh:
        log(f'Log: {mo_file(ten_log + ("_gialap" if gia_lap else ""))}')
    mods = [importlib.import_module(t if '.' in t else f'giai_doan.{t}') for t in ds_ten]
    can_cam = any(_lop(m).CAN_CAMERA for m in mods)
    if xe is None:
        if gia_lap:
            from .gia_lap import XeGiaLap
            xe = XeGiaLap(can_camera=can_cam, nhanh=nhanh)
        else:
            from .ros_io import XeThat
            xe = XeThat(can_camera=can_cam)
    imu = IMU()
    xe.gan_imu(imu)
    xem = None
    if xem_cong:
        from .xem import Xem
        xem = Xem(xem_cong)
    ctx = NguCanh(xe, imu, xem, gia_lap)
    bag = _ghi_bag(ten_log) if (ghi_bag and not gia_lap) else None
    ma = 0
    try:
        log(f'Chạy: {" -> ".join(ds_ten)} | {"GIẢ LẬP" if gia_lap else "XE THẬT"}')
        thieu = cho_cam_bien(xe, ['imu', 'scan', 'odom'] + (['camera'] if can_cam else []))
        if thieu:
            log(f'LỖI: không nhận được cảm biến {thieu} sau 5 s. Kiểm `docker ps`, crc_bringup.')
            return 2, ctx
        ctx.du_lieu['imu_hieu_chuan'] = hieu_chuan_imu(xe, imu)
        i = 0
        while i < len(mods):
            gd = _lop(mods[i])()
            log(f'==> VÀO {gd.TEN}')
            kq = chay_mot(gd, ctx)
            log(f'<== RA  {gd.TEN}: {kq.ly_do} (sau {ctx.thoi_gian():.1f} s, {ctx.quang_duong():.2f} m)')
            ctx.du_lieu.setdefault('lich_su', []).append((gd.TEN, kq.ly_do, round(ctx.thoi_gian(), 2)))
            if kq.dung_han:
                ma = 1
                break
            if kq.tiep and kq.tiep in ds_ten:
                i = ds_ten.index(kq.tiep)
            elif kq.tiep:
                log(f'(giai đoạn kế là {kq.tiep}, không nằm trong lượt này -> kết thúc)')
                break
            else:
                i += 1
        else:
            log('XONG mọi giai đoạn trong lượt này.')
    finally:
        if ctx.nhan_nen is not None:
            ctx.nhan_nen.chay = False
        xe.dung()
        if bag is not None:
            bag.send_signal(signal.SIGINT)
            try:
                bag.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bag.kill()
        xe.dong()
        if xem is not None:
            xem.dong()
        log('Đã gửi lệnh dừng xe (v = 0, w = 0).')
    return ma, ctx


class CongCu:
    """Mở xe cho một công cụ (không chạy giai đoạn). Dùng:  with CongCu('imu_test') as c: ...

    c.xe, c.imu, c.xem; c.dang_chay() = chưa bị Ctrl+C / dừng khẩn. Khi thoát luôn gửi v = w = 0.
    """

    def __init__(self, ten, can_camera=False, gia_lap=False, hieu_chuan=True, xem_cong=cau_hinh.CONG_XEM):
        self.ten, self.can_camera, self.gia_lap = ten, can_camera, gia_lap
        self.hieu_chuan, self.xem_cong = hieu_chuan, xem_cong

    def __enter__(self):
        _bat_tin_hieu()
        _DUNG['co'] = False
        try:
            cau_hinh.FILE_DUNG.unlink()
        except OSError:
            pass
        log(f'Log: {mo_file(self.ten + ("_gialap" if self.gia_lap else ""))}')
        if self.gia_lap:
            from .gia_lap import XeGiaLap
            self.xe = XeGiaLap(can_camera=self.can_camera)
        else:
            from .ros_io import XeThat
            self.xe = XeThat(can_camera=self.can_camera, ten_node=f'champion_{self.ten}')
        self.imu = IMU()
        self.xe.gan_imu(self.imu)
        self.xem = None
        if self.xem_cong:
            from .xem import Xem
            self.xem = Xem(self.xem_cong)
        thieu = cho_cam_bien(self.xe, ['imu', 'scan', 'odom'] + (['camera'] if self.can_camera else []))
        if thieu:
            self.__exit__(None, None, None)
            raise SystemExit(f'LỖI: không nhận được cảm biến {thieu} sau 5 s. Kiểm `docker ps`, crc_bringup.')
        self.so_imu = hieu_chuan_imu(self.xe, self.imu) if self.hieu_chuan else None
        return self

    def dang_chay(self):
        return can_dung() is None

    def __exit__(self, *_):
        try:
            self.xe.dung()
            self.xe.dong()
        finally:
            if self.xem is not None:
                self.xem.dong()
            log('Đã gửi lệnh dừng xe (v = 0, w = 0).')
        return False


def main(file):
    """Gọi từ `if __name__ == '__main__'` của mỗi file giai đoạn."""
    ten = Path(file).stem
    ap = argparse.ArgumentParser(description=f'Chạy giai đoạn {ten}')
    ap.add_argument('--tiep', action='store_true', help='chạy tiếp tới hết map sau giai đoạn này')
    ap.add_argument('--gia-lap', action='store_true', help='chạy trên xe giả lập (không cần xe)')
    ap.add_argument('--nhanh', action='store_true', help='giả lập chạy nhanh hết cỡ (không chờ giờ thật)')
    ap.add_argument('--khong-ghi', action='store_true', help='không ghi bag')
    a = ap.parse_args()
    # giai đoạn thử (vd s7_bam_vach_test) không nằm trong THU_TU: --tiep chỉ chạy nó
    ds = cau_hinh.THU_TU[cau_hinh.THU_TU.index(ten):] if (a.tiep and ten in cau_hinh.THU_TU) else [ten]
    ma, _ = chay(ds, gia_lap=a.gia_lap, ghi_bag=not a.khong_ghi, nhanh=a.nhanh)
    sys.exit(ma)
