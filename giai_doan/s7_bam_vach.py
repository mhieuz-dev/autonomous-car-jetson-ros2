#!/usr/bin/env python3
"""S7: BÁM VẠCH LIỀN BÊN PHẢI bằng PID theo KHOẢNG CÁCH từ xe tới vạch.

  d     = khoảng cách tới vạch phải, đo ở điểm X_NHIN (0.35 m) phía trước mũi xe,
          bằng bộ dò 8 bước (chung/lane.py). Đo ở đây vì camera NHÌN RÕ điểm này;
          kéo dài đường cong về sát xe (camera không thấy) thì sai góc tới 5 độ.
  e     = d - D_DICH          (e > 0: xe đang xa vạch / lệch trái -> phải quẹo PHẢI)
  kappa_lenh = -(KP*e + KI*tich_phan(e) + KD*de/dt)
  w     = v * kappa_lenh
  KP = 2 / X_NHIN^2 là luật "pure pursuit": điểm tâm làn cách X_NHIN phía trước
  nằm đúng trên cung tròn xe sẽ chạy, nên trên khúc cua tròn xe bám đúng tâm làn
  mà không cần đoán góc / độ cong.
  v     = V_BAM giảm theo |e| và độ cong vạch
Không đo được vạch (hoặc vạch đo ngoài dải hợp lý HOP_LY_GAN..XA): dời cả đường cong theo odom + IMU (giữ độ cong)
tối đa GIU_HOP_LY_S s, chạy chậm V_DU_DOAN; lâu hơn thì MẤT VẠCH: chạy chậm giữ hướng, quá MAT_VACH_DUNG s thì
đứng xoay phải tìm vạch (xem dưới).
Biển STOP: dừng STOP_DUNG_GIAY (khung, STOP_CA_MAP). STOP ĐẦU TIÊN trong S7: dừng xong thì CHẠY LẠI ĐƯỜNG ĐẨY TAY
(calib/duong_stop.csv, ghi bằng s7_stop_ghi) bằng pure pursuit trên vị trí tự tính (encoder + IMU, gốc là chỗ dừng
STOP), không nhìn vạch; hết quãng đường đã đẩy thì bám vạch lại với V_BAM_SAU. Không có file thì chỉ dừng rồi bám tiếp.
Chạy riêng: đặt xe giữa làn phải:  bash chay.sh s7_bam_vach
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.bam_vach import (D_DICH, DO_VACH, GIU_HOP_LY_S, MAT_VACH_DUNG, V_BAM, V_DU_DOAN, BamVach,  # noqa: E402,F401
                            DieuKhienLan, DoVachGiuCong, doi_duong_cong, tao_bo_do, y_psi)
from chung.duong_ghi import TEN_FILE_STOP, DuongDaGhi, goc as chuan_goc  # noqa: E402
from chung.log import log  # noqa: E402
from chung.pid import kep  # noqa: E402
from chung.ve import luoi  # noqa: E402

# ===== THAM SỐ (PID khoảng cách + bộ dò vạch nằm ở chung/bam_vach.py) =======
TIM_W = 0.6                # rad/s: tốc độ xoay phải mỗi nhịp tìm vạch
TIM_XOAY_GIAY = 0.25       # xoay bấy nhiêu giây (~8,6 độ ở 0.6 rad/s) ...
TIM_DUNG_GIAY = 0.5        # ... rồi đứng yên bấy nhiêu giây cho camera nhìn rõ, lặp lại tới khi thấy vạch
TIM_GOC_MAX_DO = 360.0     # xoay tổng quá số độ này vẫn chưa thấy vạch thì dừng hẳn
# Nhảy sang vạch khác: bộ dò nhận vạch mới lệch > NHAY_MAX_M so với vạch đang bám VÀ sai số mới > SAI_SO_BAT_HOP_LY_M
# -> đứng yên ngay, xoay phải từng nhịp (như trên) tới khi thấy vạch phải HỢP LÝ: khoảng cách tâm xe -> vạch
# (quy về ngay cạnh xe, không đổi khi xe xoay tại chỗ) trong [VACH_PHAI_GAN_M, VACH_PHAI_XA_M], N_KHUNG_OK khung liên tiếp.
NHAY_BAT = True
NHAY_MAX_M = 0.12          # vạch đổi chỗ đột ngột hơn chừng này (m) ... (làn 0.40, vạch kế bên cách ~0.40)
SAI_SO_BAT_HOP_LY_M = 0.12 # ... và |khoảng cách - D_DICH| sau khi nhảy lớn hơn chừng này (m) thì coi là bám nhầm vạch
VACH_PHAI_GAN_M = 0.10     # vạch phải hợp lý: cách tâm xe từ ...
VACH_PHAI_XA_M = 0.30      # ... tới chừng này (m)
N_KHUNG_OK = 2
THOI_GIAN_TOI_DA = 180.0
# ===== CHẠY LẠI ĐƯỜNG ĐẨY TAY SAU BIỂN STOP (ghi bằng s7_stop_ghi -> calib/duong_stop.csv) =====
PHAT_SAU_STOP = True       # True: STOP ĐẦU TIÊN trong S7 dừng xong thì chạy lại đường đẩy tay rồi bám vạch tiếp
PHAT_V = 0.15              # m/s khi chạy lại đường đẩy tay
PHAT_NHIN_TRUOC_M = 0.20   # pure pursuit: nhắm điểm trên đường ghi phía trước bấy nhiêu mét
PHAT_W_MAX = 1.0           # rad/s
V_BAM_SAU = 0.45           # m/s bám vạch SAU KHI chạy hết đường đẩy tay (trước STOP dùng V_BAM)
# ===========================================================================


def so_lan_stop(ctx):
    """Số lần đã dừng ở STOP từ đầu map (bộ đếm chung chung/bien_stop.DungStop)."""
    d = getattr(ctx, 'dung_stop', None)
    return 0 if d is None else d.so_lan


class GiaiDoan(khung.GiaiDoanCoSo):
    TEN = 's7_bam_vach'
    HZ = 20
    CAN_CAMERA = True
    PHAT_STOP = True                     # lớp con tắt: s7_stop_ghi (đang ghi đường), s7_phat_lai (đường ghi riêng)

    def __init__(self):
        self.THOI_GIAN_TOI_DA = THOI_GIAN_TOI_DA
        self.dk = None
        self.tim = None                  # đang xoay tìm vạch: {'t', 'goc', 'yaw', 'vi': 'mat' | 'nhay', 'ok'}
        self._so_khung = 0
        self.v_bam = V_BAM
        self.duong_stop = None           # đường đẩy tay sau STOP (None = gặp STOP chỉ dừng rồi bám tiếp)
        self.stop0 = 0                   # số lần STOP lúc vào S7: STOP mới trong S7 là khi bộ đếm vượt số này
        self.phat = None                 # đang chạy lại: {'s0', 'yaw0', 's', 'x', 'y', 'lech'}
        self.phat_xong = False

    @property
    def bam(self):
        return None if self.dk is None else self.dk.bam

    def bat_dau(self, ctx):
        self.dk = DieuKhienLan(ctx.gia_lap)
        log(f'S7: bộ dò vạch bird\'s eye {self.bam.do.be.so_cot}x{self.bam.do.be.so_hang} px, '
            f'đích cách vạch phải {D_DICH:.2f} m')
        self.stop0 = so_lan_stop(ctx)
        if PHAT_SAU_STOP and self.PHAT_STOP:
            f = cau_hinh.CALIB / TEN_FILE_STOP
            try:
                self.duong_stop = DuongDaGhi(f)
                log(f'S7: STOP đầu tiên dừng xong thì chạy lại đường đẩy tay {self.duong_stop.dai:.2f} m '
                    f'({len(self.duong_stop.s)} mẫu), rồi bám vạch {V_BAM_SAU:.2f} m/s')
            except (OSError, ValueError) as e:
                log(f'S7: không đọc được {f} ({e}) -> gặp STOP chỉ dừng rồi bám vạch tiếp')

    def _phat_sau_stop(self, ctx):
        """STOP đầu tiên trong S7 vừa dừng xong -> chạy lại đường đẩy tay. Trả (v, w) khi đang chạy lại, None = bám vạch."""
        if self.duong_stop is None or self.phat_xong:
            return None
        if self.phat is None:
            d = getattr(ctx, 'dung_stop', None)
            if d is None or d.so_lan <= self.stop0 or ctx.xe.gio() < d.den:
                return None
            self.phat = {'s0': ctx.quang_duong(), 'yaw0': ctx.imu.yaw, 's': 0.0, 'x': 0.0, 'y': 0.0, 'lech': 0.0}
            log(f'S7: hết dừng STOP -> CHẠY LẠI ĐƯỜNG ĐẨY TAY {self.duong_stop.dai:.2f} m '
                f'(hướng gốc {math.degrees(ctx.imu.yaw):+.1f} độ)')
        lenh = self._lenh_phat(ctx)
        if lenh is not None:
            return lenh
        log(f'S7: hết đường đẩy tay ({self.duong_stop.dai:.2f} m) -> bám vạch tiếp, {V_BAM_SAU:.2f} m/s')
        self.phat, self.phat_xong, self.v_bam, self.tim = None, True, V_BAM_SAU, None
        self.dk.bam.do.dat_lai()
        self.dk.dat_lai()
        return None

    def _lenh_phat(self, ctx):
        """Pure pursuit trên đường đẩy tay, vị trí tự tính = quãng đường encoder + hướng IMU (giống lúc ghi).
        Lệch ngang được kéo về chứ không cộng dồn như chỉ giữ hướng. None = đã hết đường."""
        p, duong = self.phat, self.duong_stop
        s = ctx.quang_duong() - p['s0']
        h = chuan_goc(ctx.imu.yaw - p['yaw0'])
        ds = s - p['s']
        p['x'] += ds * math.cos(h)
        p['y'] += ds * math.sin(h)
        p['s'] = s
        if s >= duong.dai:
            return None
        sx, sy = duong.diem(s)
        p['lech'] = math.hypot(sx - p['x'], sy - p['y'])
        nhin = s + PHAT_NHIN_TRUOC_M
        tx, ty = duong.diem(min(nhin, duong.dai))
        if nhin > duong.dai:                       # cuối đường: kéo dài theo hướng cuối để vẫn có điểm nhắm
            hc = duong.huong(duong.dai)
            tx, ty = tx + (nhin - duong.dai) * math.cos(hc), ty + (nhin - duong.dai) * math.sin(hc)
        dx, dy = tx - p['x'], ty - p['y']
        alpha = chuan_goc(math.atan2(dy, dx) - h)
        return PHAT_V, kep(2.0 * PHAT_V * math.sin(alpha) / max(0.05, math.hypot(dx, dy)), -PHAT_W_MAX, PHAT_W_MAX)

    def _xoay_tim(self, ctx, dt):
        """Một nhịp tìm: xoay phải TIM_XOAY_GIAY rồi đứng TIM_DUNG_GIAY. Trả (v, w) hoặc KetThuc nếu xoay quá nhiều."""
        t = self.tim
        t['goc'] += abs(math.atan2(math.sin(ctx.imu.yaw - t['yaw']), math.cos(ctx.imu.yaw - t['yaw'])))
        t['yaw'] = ctx.imu.yaw
        if math.degrees(t['goc']) > TIM_GOC_MAX_DO:
            return khung.KetThuc(f'xoay phải {TIM_GOC_MAX_DO:.0f} độ vẫn không thấy vạch', dung_han=True)
        t['t'] = (t['t'] + dt) % (TIM_XOAY_GIAY + TIM_DUNG_GIAY)
        return 0.0, (-TIM_W if t['t'] < TIM_XOAY_GIAY else 0.0)

    def buoc(self, ctx, dt):
        lenh = self._phat_sau_stop(ctx)
        if lenh is not None:
            return lenh
        v, w, het = self.dk.buoc(ctx, dt, self.v_bam, debug=ctx.xem is not None and ctx.xem.dang_xem('vach'))
        kq = self.dk.bam.kq
        moi = self.dk.so_khung != self._so_khung
        self._so_khung = self.dk.so_khung

        # ---- nhảy sang vạch khác bất hợp lý: đứng yên ngay, xoay phải tìm lại vạch phải ----
        if (NHAY_BAT and moi and self.tim is None and kq.nhay_m > NHAY_MAX_M
                and abs(kq.khoang_cach_truoc - D_DICH) > SAI_SO_BAT_HOP_LY_M):
            log(f'S7: NHẢY VẠCH {kq.nhay_m * 100:.0f} cm, cách vạch mới {kq.khoang_cach_truoc:.2f} m '
                f'(đích {D_DICH:.2f}) -> ĐỨNG YÊN, xoay phải tìm lại vạch phải')
            self.tim = {'t': 0.0, 'goc': 0.0, 'yaw': ctx.imu.yaw, 'vi': 'nhay', 'ok': 0}
            self.dk.bam.do.dat_lai()           # quên vạch nhầm, dò lại từ đầu
            self.dk.dat_lai()
            return 0.0, 0.0
        if self.tim is not None and self.tim['vi'] == 'nhay':
            if moi:
                hop_ly = kq.thay and kq.do_moi and VACH_PHAI_GAN_M <= kq.khoang_cach <= VACH_PHAI_XA_M
                self.tim['ok'] = self.tim['ok'] + 1 if hop_ly else 0
            if self.tim['ok'] >= N_KHUNG_OK:
                log(f'S7: THẤY LẠI VẠCH PHẢI cách {kq.khoang_cach:.2f} m sau khi xoay phải '
                    f'{math.degrees(self.tim["goc"]):.0f} độ -> bám vạch tiếp')
                self.tim = None
                self.dk.dat_lai()
                return v, w
            return self._xoay_tim(ctx, dt)

        # ---- mất vạch quá MAT_VACH_DUNG: đứng yên, xoay phải từng nhịp tới khi thấy vạch ----
        if not het:
            if self.tim is not None:
                log(f'S7: THẤY LẠI VẠCH sau khi xoay phải {math.degrees(self.tim["goc"]):.0f} độ -> bám vạch tiếp')
                self.tim = None
            return v, w
        if self.tim is None:
            self.tim = {'t': 0.0, 'goc': 0.0, 'yaw': ctx.imu.yaw, 'vi': 'mat', 'ok': 0}
            log(f'S7: MẤT VẠCH {MAT_VACH_DUNG:.0f} s -> dừng, xoay phải từng chút tìm vạch')
        return self._xoay_tim(ctx, dt)

    def trang_thai(self):
        if self.dk is None:
            return ''
        if self.phat is not None:
            p = self.phat
            return (f'CHẠY LẠI ĐƯỜNG ĐẨY TAY {p["s"]:.2f}/{self.duong_stop.dai:.2f} m | '
                    f'lệch đường ghi {p["lech"] * 100:.0f} cm')
        return (f'TÌM VẠCH (đã xoay phải {math.degrees(self.tim["goc"]):.0f} độ) | ' if self.tim else '') + self.dk.mo_ta()

    def ve(self, ctx):
        if not ctx.xem.dang_xem('vach'):
            ctx.xem.anh.setdefault('vach', None)     # chỉ báo là có hình, chưa ai xem thì khỏi vẽ
            return
        if self.bam and self.bam.anh:
            ctx.xem.dat_anh('vach', luoi(list(self.bam.anh.items())))
        else:
            ctx.xem.dat_anh('vach', luoi([('chưa có khung debug', None)]))


if __name__ == '__main__':
    khung.main(__file__)
