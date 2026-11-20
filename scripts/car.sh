#!/usr/bin/env bash
# CHAMPION - chay 1 script tren XE (Jetson) bang 1 lenh, log hien ngay trong terminal nay.
#
#   bash scripts/car.sh <ten> [tham so]   ten = run_car | imu_test | imu_slope | pid_test
#                                        | lidar_test | camera_test | onnx_test
#                                        | line_debug | jetson_load   (xem: bash scripts/car.sh danh_sach)
#   bash scripts/car.sh danh_sach         liet ke moi script
#   bash scripts/car.sh dung              DUNG KHAN xe (moi script champion dang chay tren xe)
#   bash scripts/car.sh keo               keo logs/ (log + bag) va calib_new/ tu xe ve laptop
#   bash scripts/car.sh nhan_calib        chep calib_new/camera.yaml, extrinsics.yaml -> config/ (sau khi duyet)
#   bash scripts/car.sh day               chi chep code len xe
#   bash scripts/car.sh ssh               mo terminal tren xe
# Nhan Ctrl+C trong luc chay = dung xe (Ctrl+C di qua ssh toi script tren xe).
# Doi IP xe:  CRC_HOST=jetson@<ip> bash scripts/car.sh ...
set -u
HOST=${CRC_HOST:-jetson@jetson.local}
CONTAINER=${CRC_CONTAINER:-crc_car}
GOC=$(cd -- "$(dirname -- "$0")/.." && pwd)
DICH=crc_ws/src/champion          # tren Jetson; trong container crc_car la /ws/src/champion
SSH="ssh -o ConnectTimeout=6"

day() {
  $SSH "$HOST" "mkdir -p ~/$DICH" || { echo "!! Khong ssh duoc toi $HOST (xe bat chua? cung Wi-Fi chua?)"; exit 1; }
  rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
    --exclude logs --exclude calib_new --exclude docs --exclude 'models/*.engine' \
    "$GOC/" "$HOST:$DICH/" || { echo "!! rsync loi"; exit 1; }
}

bat_trt() {  # may chu TensorRT tren Jetson (NGOAI container, GPU): nhan bien/den nhanh; da chay roi thi thoi
  $SSH "$HOST" "cd ~/$DICH && if [ -f models/best.engine ] && ! pgrep -f '^python3 -u tools/trt_server.py' >/dev/null; then (nohup python3 -u tools/trt_server.py > /tmp/trt_server.log 2>&1 < /dev/null &); echo '== Bat may chu TensorRT (GPU)'; fi" < /dev/null
}

dung() {
  $SSH "$HOST" "docker exec $CONTAINER bash -c 'touch /tmp/champion_stop; pkill -INT -f \"python3 -u /ws/src/champion/\"'" \
    >/dev/null 2>&1 && echo "== Da gui lenh DUNG toi xe" || echo "!! Khong gui duoc lenh dung (mat ket noi?) -> nhac xe len!"
}

dong_bo_pid() {  # ghi PID banh trong tools/pid_test.py vao ~/crc_car.env tren xe, khac thi bringup lai
  local KV; KV=$(python3 "$GOC/tools/pid_test.py" --in-env | tr "\n" " ")
  [[ "$KV" == *CRC_FF_GAIN_LEFT=* ]] || { echo "!! Loi doc tham so pid_test.py"; exit 1; }
  echo "== PID banh trong pid_test.py: $(echo $KV)"
  $SSH "$HOST" "bash -s" <<REMOTE
set -e
F=~/crc_car.env
DOI=0
for kv in $KV; do
  k=\${kv%%=*}; v=\${kv#*=}
  cu=\$(grep -E "^\$k=" \$F | tail -1 | cut -d= -f2)
  if [ "\$cu" != "\$v" ]; then
    [ \$DOI = 0 ] && cp \$F \$F.bak_champion_\$(date +%Y%m%d_%H%M%S)
    DOI=1
    if grep -qE "^\$k=" \$F; then sed -i "s/^\$k=.*/\$k=\$v/" \$F; else echo "\$k=\$v" >> \$F; fi
    echo "   \$k: \${cu:-(chua co)} -> \$v"
  fi
done
if [ \$DOI = 1 ]; then
  echo "== Da ghi ~/crc_car.env (ban cu .bak_champion_*). Khoi dong lai driver..."
  bash ~/crc_car/scripts/run_car.sh bringup >/dev/null 2>&1 || docker restart crc_bringup >/dev/null
  sleep 12
  echo "== Driver da chay lai voi so moi"
else
  echo "== ~/crc_car.env da dung so nay, khong can khoi dong lai driver"
fi
REMOTE
}

tim() {  # ten -> duong dan tuong doi
  for f in "tools/$1.py" "$1.py"; do
    [[ -f "$GOC/$f" ]] && { echo "$f"; return 0; }
  done
  return 1
}

case "${1:-}" in
  ""|-h|--help) sed -n '2,16p' "$0"; exit 0 ;;
  danh_sach)
    for f in run_car.py tools/*.py; do
      printf '  %-18s %s\n' "$(basename "$f" .py)" "$(sed -n '2p' "$GOC/$f" | sed 's/^"""//')"
    done; exit 0 ;;
  dung) dung; exit 0 ;;
  day) day; echo "== Da chep code len $HOST:~/$DICH"; exit 0 ;;
  ssh) exec $SSH -t "$HOST" ;;
  keo)
    mkdir -p "$GOC/logs" "$GOC/calib_new"
    rsync -az "$HOST:$DICH/logs/" "$GOC/logs/" && rsync -az "$HOST:$DICH/calib_new/" "$GOC/calib_new/"
    echo "== Da keo ve: $GOC/logs  $GOC/calib_new"; ls -t "$GOC/logs" | head -5; exit 0 ;;
  nhan_calib)
    for f in camera.yaml extrinsics.yaml bev.yaml; do
      if [[ -f "$GOC/calib_new/$f" ]]; then cp -v "$GOC/calib_new/$f" "$GOC/config/$f"; fi
    done
    echo "== Nho commit calib moi:  git add config && git commit"; exit 0 ;;
esac

TEN=$1; shift
REL=$(tim "$TEN") || { echo "!! Khong co script '$TEN'. Xem: bash scripts/car.sh danh_sach"; exit 1; }
NOI=$(grep -m1 -oE 'CHAY_TREN: (host|laptop)' "$GOC/$REL" | awk '{print $2}')

if [[ "$NOI" == laptop ]]; then
  exec python3 -u "$GOC/$REL" "$@"
fi
echo "== Chep code len xe ($HOST)..."
day
[[ "$TEN" == pid_test ]] && dong_bo_pid
[[ "$NOI" != host ]] && bat_trt
echo "== Chay $REL $* tren xe. Ctrl+C = DUNG XE."
if [[ "$NOI" == host ]]; then
  $SSH -tt "$HOST" "cd ~/$DICH && python3 -u $REL $*"
else
  $SSH -tt "$HOST" "docker exec -it $CONTAINER bash /ws/src/champion/scripts/in_container.sh $REL $*"
  MA=$?
  dung >/dev/null 2>&1      # chac chan khong con gi gui /cmd_vel (vo hai neu script da tu dung)
  echo "== Ket thuc (ma $MA). Log + bag nam tren xe; keo ve: bash scripts/car.sh keo"
fi
