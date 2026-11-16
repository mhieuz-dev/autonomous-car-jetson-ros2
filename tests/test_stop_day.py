"""S7 STOP: s7_stop_ghi bám vạch, gặp STOP thì đẩy tay ghi đường; S7 chính thức (s7_bam_vach) gặp STOP đầu tiên thì
dừng, chạy lại đúng đường đó rồi bám vạch tiếp với V_BAM_SAU."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chung import cau_hinh, khung  # noqa: E402
from chung.duong_ghi import DuongDaGhi, GhiDuong  # noqa: E402
from chung.gia_lap import XeGiaLap  # noqa: E402

T_STOP = (2.0, 3.0)


def _xe():
    xe = XeGiaLap(can_camera=True, nhanh=True)
    xe.x, xe.y, xe.yaw = 5.0, 1.6, math.pi / 2
    xe.t_xanh = 0.0
    xe.den_lich = [(T_STOP[0], T_STOP[1], 'stop', 90)]
    return xe


def _bat_log(monkeypatch, module):
    dong = []
    goc = module.log
    monkeypatch.setattr(module, 'log', lambda s: (dong.append(s), goc(s)))
    return dong


def test_ghi_khi_stop_roi_s7_chinh_thuc_chay_lai_dung_duong_va_bam_vach_tiep(tmp_path, monkeypatch):
    import giai_doan.s7_stop_ghi as G
    monkeypatch.setattr(cau_hinh, 'CALIB_MOI', tmp_path)
    monkeypatch.setattr(cau_hinh, 'CALIB', tmp_path)

    # ---- lượt GHI: tới STOP, dừng xong thì "đẩy tay" một cung trái rồi đoạn thẳng ----
    xe = _xe()
    day = {'t0': None, 'vet': []}
    ngu = xe.ngu

    def ngu_day(dt):
        gd = day.get('gd')
        if gd is not None and gd.ghi is not None:
            if day['t0'] is None:
                day['t0'] = xe.t
            t = xe.t - day['t0']
            xe.v_lenh, xe.w_lenh = (0.15, 0.5) if t < 8.0 else (0.15, 0.0)
            day['vet'].append((xe.x, xe.y))
        return ngu(dt)
    xe.ngu = ngu_day
    goc_init = G.GiaiDoan.__init__

    def init(self):
        goc_init(self)
        day['gd'] = self
    monkeypatch.setattr(G.GiaiDoan, '__init__', init)
    monkeypatch.setattr(G, 'THOI_GIAN_TOI_DA', 16.0)
    dong_ghi = _bat_log(monkeypatch, G.S7)
    khung.chay(['s7_stop_ghi'], gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    assert not any('CHẠY LẠI ĐƯỜNG ĐẨY TAY' in s for s in dong_ghi), 'lượt ghi không được chạy lại đường cũ'
    d = DuongDaGhi(tmp_path / G.TEN_FILE)
    assert d.dai > 1.2, d.dai
    assert 100 < math.degrees(d.huong(0.6)) < 130                # cung trái 0,5 rad/s: 0,6 m ~ 4 s ~ 115 độ
    x0, y0 = day['vet'][0]
    x1, y1 = day['vet'][-1]
    monkeypatch.undo()

    # ---- lượt CHẠY: S7 chính thức, cùng chỗ xuất phát, tới STOP thì chạy lại đường ghi rồi bám vạch tiếp ----
    import giai_doan.s7_bam_vach as S7
    monkeypatch.setattr(cau_hinh, 'CALIB', tmp_path)
    xe = _xe()
    vet, v_sau = [], []
    goc_gui = xe.gui

    def gui(v, w):
        vet.append((xe.t, xe.x, xe.y))
        goc_gui(v, w)
    xe.gui = gui
    monkeypatch.setattr(S7, 'THOI_GIAN_TOI_DA', 30.0)
    dong = _bat_log(monkeypatch, S7)
    goc_dk = S7.DieuKhienLan.buoc

    def dk_buoc(self, ctx, dt, v_bam=S7.V_BAM, debug=False):
        if any('hết đường đẩy tay' in s for s in dong):
            v_sau.append(v_bam)
        return goc_dk(self, ctx, dt, v_bam, debug)
    monkeypatch.setattr(S7.DieuKhienLan, 'buoc', dk_buoc)
    khung.chay(['s7_bam_vach'], gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    assert any('CHẠY LẠI ĐƯỜNG ĐẨY TAY' in s for s in dong), dong
    assert any('hết đường đẩy tay' in s for s in dong), dong
    # đường đẩy giả kết thúc ngoài sa bàn nên không kiểm tốc độ thật, chỉ kiểm tốc độ bám vạch được truyền vào
    assert v_sau and all(v == S7.V_BAM_SAU for v in v_sau), set(v_sau)
    gan = min(math.hypot(x - x1, y - y1) for _, x, y in vet)   # điểm xe đi qua gần nhất với điểm cuối lượt đẩy
    print(f'chạy lại: cách điểm cuối lượt đẩy {gan * 100:.1f} cm (đẩy {d.dai:.2f} m)')
    assert gan < 0.08, f'lệch điểm cuối {gan:.3f} m'
    assert math.hypot(vet[0][1] - 5.0, vet[0][2] - 1.6) < 0.5 and math.hypot(x0 - 5.0, y0 - 1.6) < 1.0


def test_khong_co_duong_day_tay_thi_stop_chi_dung_roi_bam_vach(tmp_path, monkeypatch):
    import giai_doan.s7_bam_vach as S7
    monkeypatch.setattr(cau_hinh, 'CALIB', tmp_path)               # thư mục trống: không có duong_stop.csv
    monkeypatch.setattr(S7, 'THOI_GIAN_TOI_DA', 8.0)
    dong = _bat_log(monkeypatch, S7)
    xe = _xe()
    khung.chay(['s7_bam_vach'], gia_lap=True, nhanh=True, xem_cong=None, xe=xe)
    assert any('gặp STOP chỉ dừng rồi bám vạch tiếp' in s for s in dong), dong
    assert not any('CHẠY LẠI ĐƯỜNG ĐẨY TAY' in s for s in dong)
    assert xe.va_cham is None


def test_full_map_s1_toi_s7_stop_o_s7_chay_lai_duong_day_tay(tmp_path, monkeypatch):
    """Như lúc thi: bấm Tới hết ở S1, chạy hết S1 -> S7; biển STOP hiện ra khi đang ở S7."""
    from test_gia_lap import chay_va_do
    import giai_doan.s7_bam_vach as S7
    g = GhiDuong()                                   # sa bàn giả khác đường thật: dùng đường đẩy thẳng 0,6 m
    for i in range(61):
        g.cap_nhat(i * 0.01, 0.0, 0.0, 0.0, 0.15, float(i))
    f = tmp_path / 'duong_stop.csv'
    g.luu(f)
    monkeypatch.setattr(S7, 'DuongDaGhi', lambda *_: DuongDaGhi(f))
    goc_bd = S7.GiaiDoan.bat_dau

    def bat_dau(self, ctx):
        goc_bd(self, ctx)
        ctx.xe.den_lich = [(ctx.xe.t + 1.0, ctx.xe.t + 2.0, 'stop', 90)]
    monkeypatch.setattr(S7.GiaiDoan, 'bat_dau', bat_dau)
    dong = _bat_log(monkeypatch, S7)
    ma, ctx, xe, ghi = chay_va_do()
    assert [g for g, _, _ in ctx.du_lieu['lich_su']] == cau_hinh.THU_TU
    assert cau_hinh.THU_TU[-1] == 's7_bam_vach'
    assert xe.va_cham is None, f'va chạm {xe.va_cham}'
    assert any('CHẠY LẠI ĐƯỜNG ĐẨY TAY 0.60 m' in s for s in dong), dong
    assert any('hết đường đẩy tay' in s for s in dong), dong
