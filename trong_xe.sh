#!/usr/bin/env bash
# Chay 1 script Champion BEN TRONG container crc_car (chay.sh goi file nay).
# Go tay tren Jetson (log hien ngay terminal do):
#   docker exec -it crc_car bash /ws/src/champion/trong_xe.sh xe_tu_hanh.py
#   docker exec -it crc_car bash /ws/src/champion/trong_xe.sh cong_cu/lidar_test.py
source /opt/ros/humble/setup.bash
[[ -f /engines/fastdds_mtu.xml ]] && export FASTRTPS_DEFAULT_PROFILES_FILE=/engines/fastdds_mtu.xml
# Tat MOI thu khac dang gui /cmd_vel: code cu (crc_solution, imu_yaw_loop, ...) va lan chay truoc
pkill -INT -f "lib/crc_solution/solution" 2>/dev/null
pkill -INT -f "lib/crc_solutionn/" 2>/dev/null
pkill -INT -f "run_solution_laptop.py" 2>/dev/null
pkill -INT -f "imu_yaw_loop.py" 2>/dev/null
pkill -INT -f "python3 -u /ws/src/champion/" 2>/dev/null
pkill -INT -f "ros2 bag record -o /ws/src/champion" 2>/dev/null
sleep 1
cd /ws/src/champion || exit 1
REL=$1; shift
exec python3 -u /ws/src/champion/"$REL" "$@"
