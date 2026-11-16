"""Chương trình lái tự hành: mỗi nhịp nhận biết tình huống -> chọn hành vi theo ưu tiên -> ra lệnh (v, w).

Phần chung của mọi script nằm ở chung/khung.py: dừng khẩn (Ctrl+C, file /tmp/champion_dung, mất cảm biến),
hiệu chuẩn IMU, giới hạn gia tốc, vòng gyro, chặn sát mũi, log, ghi bag, trang xem hình.
"""
import math

from chung import khung
from chung.log import log
from chung.ve import lidar_tren_xuong, luoi

from . import tham_so as TS
from .hanh_vi import tao_ds
from .nhan_biet import NhanBiet, nguong_nguoi


class ChuongTrinh(khung.GiaiDoanCoSo):
    TEN = 'tu_hanh'
    HZ = TS.HZ
    CAN_CAMERA = True

    def __init__(self):
        self.THOI_GIAN_TOI_DA = TS.THOI_GIAN_TOI_DA
        self.nb = None
        self.ds = []
        self.dang = None
        self.th = None

    def bat_dau(self, ctx):
        self.nb = NhanBiet(ctx)
        self.ds = tao_ds(self.nb.bam)
        ctx.du_lieu['hanh_vi'] = []         # [(giờ, tên)] mỗi lần đổi hành vi, để xem lại / test
        ctx.du_lieu['doc'] = []             # [(giờ, trạng thái dốc)] mỗi lần đổi
        log('Tự hành, ưu tiên: ' + ' > '.join(hv.TEN for hv in self.ds))

    def buoc(self, ctx, dt):
        th = self.th = self.nb.cap_nhat(ctx, dt)
        if not ctx.du_lieu['doc'] or ctx.du_lieu['doc'][-1][1] != th.doc:
            ctx.du_lieu['doc'].append((round(th.t, 2), th.doc))
        muon = [hv for hv in self.ds if hv.muon(th)]     # hỏi hết để bộ đếm giờ của từng hành vi luôn đúng
        hv = muon[0]
        if hv is not self.dang:
            cu = self.dang.TEN if self.dang else '-'
            log(f'HÀNH VI: {cu} -> {hv.TEN}' + (f' ({hv.ly_do})' if hv.ly_do else ''))
            ctx.du_lieu['hanh_vi'].append((round(th.t, 2), hv.TEN))
            self.dang = hv
            hv.bat_dau(th)
        lenh = hv.lenh(th, dt)
        if hv.ket_thuc:
            return khung.KetThuc(hv.ket_thuc, dung_han=True)
        return lenh

    def ket_thuc(self, ctx, ly_do):
        if self.nb is not None:
            self.nb.dung()

    def trang_thai(self):
        th, hv = self.th, self.dang
        if th is None or hv is None:
            return ''
        ham = 'TRONG HẦM' if th.trong_ham else f'cổng hầm {th.cong_ham}'
        return f'{hv.TEN} | dốc {th.doc} {th.pitch_do:+.1f} độ | {ham} | {hv.mo_ta()}'

    def ve(self, ctx):
        th = self.th
        if th is None:
            return
        d = nguong_nguoi()
        ctx.xem.dat_anh('lidar', lidar_tren_xuong(
            th.scan, non=[(0, math.degrees(math.atan2(TS.NUA_RONG_M, d)), d, (0, 0, 255) if th.nguoi else (0, 255, 0))],
            chu=self.dang.TEN if self.dang else ''))
        if ctx.xem.dang_xem('vach'):
            anh = self.nb.bam.anh
            ctx.xem.dat_anh('vach', luoi(list(anh.items())) if anh else luoi([('chưa có khung debug', None)]))
        else:
            ctx.xem.anh.setdefault('vach', None)     # chỉ báo là có hình, chưa ai xem thì khỏi vẽ
