#!/usr/bin/env bash
# CHAMPION - chay 1 script tren XE (Jetson) bang 1 lenh, log hien ngay trong terminal nay.
#
#   bash scripts/car.sh <ten> [tham so]   ten = run_car | imu_test | imu_slope | pid_test
#                                         | lidar_test | camera_test | onnx_test
#                                         | line_debug | jetson_load   (xem: bash scripts/car.sh list)
#   bash scripts/car.sh list              liet ke moi script
#   bash scripts/car.sh stop              DUNG KHAN xe (moi script champion dang chay tren xe)
#   bash scripts/car.sh pull              keo logs/ (log + bag) va calib_new/ tu xe ve laptop
#   bash scripts/car.sh accept_calib      chep calib_new/camera.yaml, extrinsics.yaml -> config/ (sau khi duyet)
#   bash scripts/car.sh push              chi chep code len xe
#   bash scripts/car.sh ssh               mo terminal tren xe
# Nhan Ctrl+C trong luc chay = dung xe (Ctrl+C di qua ssh toi script tren xe).
# Doi IP xe:  CRC_HOST=jetson@<ip> bash scripts/car.sh ...
set -u
HOST=${CRC_HOST:-jetson@jetson.local}
CONTAINER=${CRC_CONTAINER:-crc_car}
ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
DEST=crc_ws/src/champion          # tren Jetson; trong container crc_car la /ws/src/champion
SSH="ssh -o ConnectTimeout=6"

push() {
  $SSH "$HOST" "mkdir -p ~/$DEST" || { echo "!! Khong ssh duoc toi $HOST (xe bat chua? cung Wi-Fi chua?)"; exit 1; }
  rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
    --exclude logs --exclude calib_new --exclude docs --exclude 'models/*.engine' \
    "$ROOT/" "$HOST:$DEST/" || { echo "!! rsync loi"; exit 1; }
}

start_trt() {  # may chu TensorRT tren Jetson (NGOAI container, GPU): nhan bien/den nhanh; da chay roi thi thoi
  $SSH "$HOST" "cd ~/$DEST && if [ -f models/best.engine ] && ! pgrep -f '^python3 -u tools/trt_server.py' >/dev/null; then (nohup python3 -u tools/trt_server.py > /tmp/trt_server.log 2>&1 < /dev/null &); echo '== Bat may chu TensorRT (GPU)'; fi" < /dev/null
}

stop_car() {
  $SSH "$HOST" "docker exec $CONTAINER bash -c 'touch /tmp/champion_stop; pkill -INT -f \"python3 -u /ws/src/champion/\"'" \
    >/dev/null 2>&1 && echo "== Da gui lenh DUNG toi xe" || echo "!! Khong gui duoc lenh dung (mat ket noi?) -> nhac xe len!"
}

sync_pid() {  # ghi PID banh trong tools/pid_test.py vao ~/crc_car.env tren xe, khac thi bringup lai
  local KV; KV=$(python3 "$ROOT/tools/pid_test.py" --in-env | tr "\n" " ")
  [[ "$KV" == *CRC_FF_GAIN_LEFT=* ]] || { echo "!! Loi doc tham so pid_test.py"; exit 1; }
  echo "== PID banh trong pid_test.py: $(echo $KV)"
  $SSH "$HOST" "bash -s" <<REMOTE
set -e
F=~/crc_car.env
CHANGED=0
for kv in $KV; do
  k=\${kv%%=*}; v=\${kv#*=}
  old=\$(grep -E "^\$k=" \$F | tail -1 | cut -d= -f2)
  if [ "\$old" != "\$v" ]; then
    [ \$CHANGED = 0 ] && cp \$F \$F.bak_champion_\$(date +%Y%m%d_%H%M%S)
    CHANGED=1
    if grep -qE "^\$k=" \$F; then sed -i "s/^\$k=.*/\$k=\$v/" \$F; else echo "\$k=\$v" >> \$F; fi
    echo "   \$k: \${old:-(chua co)} -> \$v"
  fi
done
if [ \$CHANGED = 1 ]; then
  echo "== Da ghi ~/crc_car.env (ban cu .bak_champion_*). Khoi dong lai driver..."
  bash ~/crc_car/scripts/run_car.sh bringup >/dev/null 2>&1 || docker restart crc_bringup >/dev/null
  sleep 12
  echo "== Driver da chay lai voi so moi"
else
  echo "== ~/crc_car.env da dung so nay, khong can khoi dong lai driver"
fi
REMOTE
}

find_script() {  # ten -> duong dan tuong doi
  for f in "tools/$1.py" "$1.py"; do
    [[ -f "$ROOT/$f" ]] && { echo "$f"; return 0; }
  done
  return 1
}

case "${1:-}" in
  ""|-h|--help) sed -n '2,16p' "$0"; exit 0 ;;
  list)
    for f in run_car.py tools/*.py; do
      [[ "$(basename "$f")" == __init__.py || "$(basename "$f")" == control_panel.py ]] && continue
      printf '  %-18s %s\n' "$(basename "$f" .py)" "$(sed -n '2p' "$ROOT/$f" | sed 's/^"""//')"
    done; exit 0 ;;
  stop) stop_car; exit 0 ;;
  push) push; echo "== Da chep code len $HOST:~/$DEST"; exit 0 ;;
  ssh) exec $SSH -t "$HOST" ;;
  pull)
    mkdir -p "$ROOT/logs" "$ROOT/calib_new"
    rsync -az "$HOST:$DEST/logs/" "$ROOT/logs/" && rsync -az "$HOST:$DEST/calib_new/" "$ROOT/calib_new/"
    echo "== Da keo ve: $ROOT/logs  $ROOT/calib_new"; ls -t "$ROOT/logs" | head -5; exit 0 ;;
  accept_calib)
    for f in camera.yaml extrinsics.yaml bev.yaml; do
      if [[ -f "$ROOT/calib_new/$f" ]]; then cp -v "$ROOT/calib_new/$f" "$ROOT/config/$f"; fi
    done
    echo "== Nho commit calib moi:  git add config && git commit"; exit 0 ;;
esac

NAME=$1; shift
REL=$(find_script "$NAME") || { echo "!! Khong co script '$NAME'. Xem: bash scripts/car.sh list"; exit 1; }
RUNS_ON=$(grep -m1 -oE 'RUNS_ON: (host|laptop)' "$ROOT/$REL" | awk '{print $2}')

if [[ "$RUNS_ON" == laptop ]]; then
  exec python3 -u "$ROOT/$REL" "$@"
fi
echo "== Chep code len xe ($HOST)..."
push
[[ "$NAME" == pid_test ]] && sync_pid
[[ "$RUNS_ON" != host ]] && start_trt
echo "== Chay $REL $* tren xe. Ctrl+C = DUNG XE."
if [[ "$RUNS_ON" == host ]]; then
  $SSH -tt "$HOST" "cd ~/$DEST && python3 -u $REL $*"
else
  $SSH -tt "$HOST" "docker exec -it $CONTAINER bash /ws/src/champion/scripts/in_container.sh $REL $*"
  CODE=$?
  stop_car >/dev/null 2>&1      # chac chan khong con gi gui /cmd_vel (vo hai neu script da tu dung)
  echo "== Ket thuc (ma $CODE). Log + bag nam tren xe; keo ve: bash scripts/car.sh pull"
fi
