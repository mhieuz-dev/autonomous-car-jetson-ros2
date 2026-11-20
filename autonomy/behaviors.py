"""Các hành vi lái. Mỗi nhịp, driver.py hỏi mọi hành vi `muon(th)` (có cần nắm lái không), hành vi đầu tiên
theo thứ tự ưu tiên trả lời có thì ra lệnh `lenh(th, dt)` -> (v, w), hoặc (v, w, 'gap') = đặt ngay không qua
giới hạn gia tốc (dùng khi dừng).

Ưu tiên (tao_ds cuối file): an toàn trước, rồi luật giao thông, rồi địa hình, cuối cùng là bám vạch.
Hành vi không được chọn vẫn được hỏi `muon` mỗi nhịp, nên bộ đếm giờ của nó luôn đúng.
"""
import math

from core import settings, tunnel
from core.lane_follow import D_DICH, DieuKhienLan
from core.stop_sign import DungStop
from core.lidar import range_in_cone
from core.log import log
from core.pid import GiuHuong, goc

from . import params as TS

_TEN_DEN = {'do': 'ĐỎ', 'vang': 'VÀNG', 'xanh': 'XANH'}


class HanhVi:
    TEN = '?'

    def __init__(self):
        self.ly_do = ''              # vì sao muốn nắm lái (in ra log khi đổi hành vi)
        self.ket_thuc = None         # đặt chuỗi lý do khi hành vi muốn dừng hẳn lượt chạy

    def muon(self, th):
        return False

    def bat_dau(self, th):
        """Gọi khi vừa được chọn."""

    def lenh(self, th, dt):
        raise NotImplementedError

    def mo_ta(self):
        return ''


class ChoNguoi(HanhVi):
    """Người đi bộ trước mũi (LiDAR): đứng yên tới khi vùng trước trống liên tục THOI_GIAN_TRONG giây."""
    TEN = 'cho_nguoi'

    def __init__(self):
        super().__init__()
        self.t_thay = None           # lần cuối còn thấy người, None = không chờ
        self.lan = 0

    def muon(self, th):
        if th.nguoi:
            if self.t_thay is None:
                self.lan += 1
                self.ly_do = f'người cách LiDAR {th.nguoi_cach:.2f} m (lần {self.lan})'
            self.t_thay = th.t
            return True
        if self.t_thay is not None and th.t - self.t_thay < TS.THOI_GIAN_TRONG and th.doc == 'phang':
            return True
        self.t_thay = None
        return False

    def lenh(self, th, dt):
        return 0.0, 0.0, 'gap'

    def mo_ta(self):
        return f'chờ người (lần {self.lan})'


class DungDen(HanhVi):
    """Đèn đỏ / vàng đủ gần: dừng, chờ đèn xanh DEN_XANH_LIEN lần nhận liền. Chờ quá DEN_CHO_TOI_DA giây thì đi tiếp
    và bỏ qua đèn DEN_BO_QUA_GIAY giây (thà chạy còn hơn kẹt cả lượt vì nhận nhầm). Không dừng trên dốc."""
    TEN = 'dung_den'

    def __init__(self):
        super().__init__()
        self.t_dung = None
        self.xanh = 0
        self.lan = 0
        self.bo_qua_toi = -1.0

    def muon(self, th):
        if not TS.DEN_BAT or th.doc != 'phang' or th.t < self.bo_qua_toi:
            self.t_dung = None
            return False
        if self.t_dung is None:
            if th.den not in ('do', 'vang'):
                return False
            self.t_dung, self.xanh = th.t, 0
            self.lan += 1
            self.ly_do = f'đèn {_TEN_DEN[th.den]} cao {th.den_cao:.0f} px (lần {self.lan})'
            return True
        self.xanh = self.xanh + 1 if th.den == 'xanh' else 0
        if self.xanh >= TS.DEN_XANH_LIEN * max(1, int(TS.DEN_CHU_KY * TS.HZ)):
            log('ĐÈN: xanh -> đi tiếp')
            self.t_dung = None
            return False
        if th.t - self.t_dung > TS.DEN_CHO_TOI_DA:
            log(f'ĐÈN: CẢNH BÁO chờ quá {TS.DEN_CHO_TOI_DA:.0f} s chưa thấy xanh -> đi tiếp, '
                f'bỏ qua đèn {TS.DEN_BO_QUA_GIAY:.0f} s (kiểm tra nhận đèn)')
            self.t_dung, self.bo_qua_toi = None, th.t + TS.DEN_BO_QUA_GIAY
            return False
        return True

    def lenh(self, th, dt):
        return 0.0, 0.0, 'gap'

    def mo_ta(self):
        return f'chờ xanh ({self.xanh} lần xanh liền)'


class DungBienStop(HanhVi):
    """Biển STOP đủ gần: đứng yên STOP_DUNG_GIAY giây rồi đi tiếp, sau đó bỏ qua STOP STOP_BO_QUA_GIAY giây để chạy
    qua khỏi biển (luật ở core/stop_sign.py). Không dừng trên dốc."""
    TEN = 'dung_stop'

    def __init__(self):
        super().__init__()
        self.luat = DungStop()

    def muon(self, th):
        if not TS.BIEN_BAT or th.doc != 'phang':
            return False
        dung = self.luat.buoc(th.bien, th.t, 'STOP', TS.STOP_CAO_MIN_PX, TS.BIEN_TIN_MIN,
                              TS.STOP_DUNG_GIAY, TS.STOP_BO_QUA_GIAY)
        self.ly_do = f'biển STOP (lần {self.luat.so_lan})'
        return dung

    def lenh(self, th, dt):
        return 0.0, 0.0, 'gap'


class QuaHam(HanhVi):
    """Trong hầm chỉ tin LiDAR: bám tâm hành lang bằng luật lái 04a (core/tunnel.py), khúc cua thì chậm lại quẹo về
    phía mở. GO_KET_BAT: kẹt quá GO_KET_SAU giây thì lùi thẳng một đoạn rồi đi tiếp."""
    TEN = 'qua_ham'

    def __init__(self):
        super().__init__()
        self.t_ket = 0.0
        self.lui_tu = None           # quãng đường lúc bắt đầu lùi (None = không lùi)
        self.so_lan_lui = 0
        self.v_truoc = 0.0

    def muon(self, th):
        self.ly_do = 'LiDAR thấy 2 tường'
        return th.trong_ham

    def bat_dau(self, th):
        self.t_ket, self.lui_tu, self.v_truoc = 0.0, None, 0.0

    def lenh(self, th, dt):
        if TS.GO_KET_BAT:
            lui = self._go_ket(th, dt)
            if lui is not None:
                return lui
        if th.ham_cua is not None:
            v, w = th.ham_cua
        elif th.ham_pose is not None:
            v, w = tunnel.V_HAM, tunnel.lenh_lai(*th.ham_pose, tunnel.V_HAM)
        else:
            v, w = tunnel.V_HAM, 0.0
        self.v_truoc = v
        return v, w

    def _go_ket(self, th, dt):
        """Kẹt (chặn mũi, hoặc ra lệnh chạy mà bánh không lăn) quá GO_KET_SAU giây -> lùi thẳng."""
        if self.lui_tu is None:
            ket = th.chan_mui or (self.v_truoc > 0.03 and abs(th.v) < 0.015)
            self.t_ket = self.t_ket + dt if ket else 0.0
            if self.t_ket < TS.GO_KET_SAU:
                return None
            self.lui_tu = th.quang_duong
            self.so_lan_lui += 1
            log(f'HẦM: KẸT {self.t_ket:.1f} s -> lùi {TS.LUI_QUANG_DUONG} m rồi đi tiếp (lần {self.so_lan_lui})')
        sau = range_in_cone(th.scan, 180.0, 25.0)
        het_cho = sau is not None and sau < settings.LIDAR_TOI_CAN_SAU_M + TS.SAU_TRONG_M
        if th.quang_duong - self.lui_tu >= TS.LUI_QUANG_DUONG or het_cho:
            log('HẦM: lùi xong' + (' (sau sát vật)' if het_cho else '') + ', đi tiếp')
            self.lui_tu, self.t_ket, self.v_truoc = None, 0.0, 0.0
            return None
        self.v_truoc = TS.LUI_V
        return TS.LUI_V, 0.0

    def mo_ta(self):
        return 'ĐANG LÙI' if self.lui_tu is not None else ''


class QuaDoc(HanhVi):
    """Trên dốc camera chúc lên / xuống nên bird's eye đo vạch sai: giữ hướng lúc vào dốc bằng IMU, tốc độ theo
    phần dốc. Không bao giờ đứng yên hay lùi trên dốc (luật thi)."""
    TEN = 'qua_doc'

    def __init__(self):
        super().__init__()
        self.huong = GiuHuong(TS.KP_HUONG_DOC, TS.KI_HUONG_DOC, TS.KD_HUONG_DOC, TS.W_MAX_DOC)
        self.sai = 0.0

    def muon(self, th):
        self.ly_do = f'dốc {th.doc}'
        return th.doc != 'phang'

    def bat_dau(self, th):
        self.huong.reset()

    def lenh(self, th, dt):
        w, self.sai = self.huong.tinh(th.imu, th.yaw_doc, dt)
        if abs(math.degrees(self.sai)) > TS.LECH_NHIEU_DO:
            return TS.V_CHINH_HUONG, w
        v = {'len': TS.V_LEN_DOC, 'dinh': TS.V_DINH_DOC}.get(th.doc, TS.V_XUONG_DOC)
        return v, w

    def mo_ta(self):
        return f'lệch hướng {math.degrees(self.sai):+.1f} độ'


class RaHam(HanhVi):
    """Vừa ra hầm camera chưa bắt được vạch: giữ hướng lúc ra hầm, chạy chậm ít nhất RA_HAM_TROI_MIN giây; thấy vạch
    RA_HAM_SO_KHUNG_VACH khung liền (hoặc hết RA_HAM_TROI_MAX giây) thì trả lái cho bám vạch."""
    TEN = 'ra_ham'

    def __init__(self):
        super().__init__()
        self.huong = GiuHuong(1.5, 0.0, 0.10, 0.6)

    def muon(self, th):
        if th.t_ra_ham is None:
            return False
        t = th.t - th.t_ra_ham
        self.ly_do = 'vừa ra hầm'
        return t < TS.RA_HAM_TROI_MAX and (t < TS.RA_HAM_TROI_MIN or th.vach_lien < TS.RA_HAM_SO_KHUNG_VACH)

    def bat_dau(self, th):
        self.huong.reset()

    def lenh(self, th, dt):
        w, _ = self.huong.tinh(th.imu, th.yaw_ra_ham, dt)
        return TS.V_RA_HAM, w

    def mo_ta(self):
        return 'giữ hướng chờ thấy vạch'


class BamVach(HanhVi):
    """Mặc định: bám vạch phải (core/lane_follow.py). Vạch nhảy sang chỗ bất hợp lý (nhận nhầm vạch làn bên) thì đứng
    yên xoay phải tìm lại; mất vạch quá MAT_VACH_DUNG giây cũng vậy. Xoay hết TIM_GOC_MAX_DO vẫn không thấy = hết
    đường, dừng hẳn lượt chạy."""
    TEN = 'bam_vach'

    def __init__(self, bam):
        super().__init__()
        self.dk = DieuKhienLan(bam=bam)
        self.tim = None              # đang xoay tìm vạch: {'t', 'goc', 'yaw', 'vi': 'mat' | 'nhay', 'ok'}

    def muon(self, th):
        self.ly_do = ''
        return True

    def bat_dau(self, th):
        self.dk.dat_lai()
        self.tim = None

    def lenh(self, th, dt):
        v, w, het = self.dk.lenh(th.vach, th.imu, dt, TS.V_BAM_VACH, t=th.t)
        kq, moi = th.vach, th.vach_moi

        # ---- nhảy sang vạch khác bất hợp lý: đứng yên ngay, xoay phải tìm lại vạch phải ----
        if (TS.NHAY_BAT and moi and self.tim is None and kq.nhay_m > TS.NHAY_MAX_M
                and abs(kq.khoang_cach_truoc - D_DICH) > TS.SAI_SO_BAT_HOP_LY_M):
            log(f'VẠCH: nhảy {kq.nhay_m * 100:.0f} cm, cách vạch mới {kq.khoang_cach_truoc:.2f} m '
                f'(đích {D_DICH:.2f}) -> đứng yên, xoay phải tìm lại vạch phải')
            self.tim = {'t': 0.0, 'goc': 0.0, 'yaw': th.imu.yaw, 'vi': 'nhay', 'ok': 0}
            self.dk.bam.do.dat_lai()           # quên vạch nhầm, dò lại từ đầu
            self.dk.dat_lai()
            return 0.0, 0.0
        if self.tim is not None and self.tim['vi'] == 'nhay':
            if moi:
                hop_ly = kq.thay and kq.do_moi and TS.VACH_PHAI_GAN_M <= kq.khoang_cach <= TS.VACH_PHAI_XA_M
                self.tim['ok'] = self.tim['ok'] + 1 if hop_ly else 0
            if self.tim['ok'] >= TS.N_KHUNG_OK:
                log(f'VẠCH: thấy lại vạch phải cách {kq.khoang_cach:.2f} m sau khi xoay '
                    f'{math.degrees(self.tim["goc"]):.0f} độ -> bám tiếp')
                self.tim = None
                self.dk.dat_lai()
                return v, w
            return self._xoay_tim(th, dt)

        # ---- mất vạch quá MAT_VACH_DUNG: đứng yên, xoay phải từng nhịp tới khi thấy vạch ----
        if not het:
            if self.tim is not None:
                log(f'VẠCH: thấy lại vạch sau khi xoay {math.degrees(self.tim["goc"]):.0f} độ -> bám tiếp')
                self.tim = None
            return v, w
        if self.tim is None:
            self.tim = {'t': 0.0, 'goc': 0.0, 'yaw': th.imu.yaw, 'vi': 'mat', 'ok': 0}
            log('VẠCH: mất vạch quá lâu -> dừng, xoay phải từng chút tìm vạch')
        return self._xoay_tim(th, dt)

    def _xoay_tim(self, th, dt):
        """Một nhịp tìm: xoay phải TIM_XOAY_GIAY rồi đứng TIM_DUNG_GIAY."""
        t = self.tim
        t['goc'] += abs(goc(th.imu.yaw - t['yaw']))
        t['yaw'] = th.imu.yaw
        if math.degrees(t['goc']) > TS.TIM_GOC_MAX_DO:
            self.ket_thuc = f'xoay phải {TS.TIM_GOC_MAX_DO:.0f} độ vẫn không thấy vạch (hết đường)'
            return 0.0, 0.0, 'gap'
        t['t'] = (t['t'] + dt) % (TS.TIM_XOAY_GIAY + TS.TIM_DUNG_GIAY)
        return 0.0, (-TS.TIM_W if t['t'] < TS.TIM_XOAY_GIAY else 0.0)

    def mo_ta(self):
        tim = f'TÌM VẠCH (đã xoay {math.degrees(self.tim["goc"]):.0f} độ) | ' if self.tim else ''
        return tim + self.dk.mo_ta()


def tao_ds(bam):
    """Mọi hành vi, ưu tiên cao trước. bam = BamVach của lớp nhận biết (dùng chung khung camera đã dò)."""
    return [ChoNguoi(), DungDen(), DungBienStop(), QuaHam(), QuaDoc(), RaHam(), BamVach(bam)]
