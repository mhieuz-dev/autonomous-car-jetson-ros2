# Cấu hình camera và xe

| File | Nội dung |
|---|---|
| `camera.yaml` | Calib nội camera (fx 552,8, fy 551,9, cx 311,4, cy 239,4), chuyển từ `intrinsics_raw.yaml` |
| `intrinsics_raw.yaml` | File calib gốc (bàn cờ 10x7 góc trong, ô 19 mm) |
| `extrinsics.yaml` | Tư thế camera so với mặt sàn (rvec, tvec, chiều cao), sinh ra khi chọn ứng viên ở bảng điều khiển |
| `bev.yaml` | Vùng nhìn và độ phân giải bird's eye, chọn cùng lúc với `extrinsics.yaml` |
| `sign_roi.yaml` | Dải hàng pixel dùng để nhận biển báo và đèn, sửa trên bảng điều khiển |
| `fastdds_mtu.xml` | Hồ sơ Fast DDS giới hạn gói UDP 1400 byte (hotspot làm rớt gói bị phân mảnh) |

## Đo bird's eye bằng bàn cờ

1. `python3 tools/control_panel.py`, nhóm "Bird's eye": bấm **Xe** ở `birdeye_chessboard`, chốt 2-3 lần, mỗi lần đặt bàn cờ một chỗ khác.
2. Bấm **Kéo kết quả từ xe**, tick các lần tốt nhất (nhiều lần thì lấy trung vị).
3. Chọn vùng nhìn và độ phân giải, **Xem thử**, **Lưu lựa chọn**, rồi **Dùng số này** để chép vào `config/`.

Chưa có `extrinsics.yaml` thì mọi script cần camera sẽ dừng và báo thiếu. Kết quả đo chờ duyệt nằm ở `calib_new/`, thư mục này không đưa lên git.
