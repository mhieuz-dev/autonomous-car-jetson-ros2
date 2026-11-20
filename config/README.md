THU MUC CALIB CAMERA (camera moi, 10/10/2026)

  camera.yaml          calib NOI (fx 552.8, fy 551.9, cx 311.4, cy 239.4), chuyen tu intrinsics_raw.yaml
  intrinsics_raw.yaml  file goc nguoi dung dua (ban co 10x7, o 19 mm)
  extrinsics.yaml           tu the camera so voi mat san (rvec, tvec), sinh ra khi ban CHON UNG VIEN o bang dieu khien
  bev.yaml             vung nhin + do phan giai bird's eye, ban chon cung luc do
  birdeye_chessboard.py     chay tren xe: bird's eye bang ban co nam tren san, moi lan chot luu mot UNG VIEN
  birdeye_select.html    khoi giao dien chon ung vien (gan vao bang dieu khien chung)

Cach lam: python3 tools/control_panel.py -> nhom "Bird's eye": bam Xe o birdeye_chessboard, chot 2-3 lan (dat ban co cho khac),
keo ket qua, tick lan tot nhat, chon vung nhin + do phan giai, Xem thu, Luu, Dung so nay.
Chua co extrinsics.yaml thi moi script can camera dung va bao thieu. calib_new/ (ket qua cho duyet) khong len git.
