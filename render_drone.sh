#!/bin/bash
# Render the drone video in 8 parallel parts on GPUs 1 and 2, then join them.
cd "$(dirname "$0")"
mkdir -p renders/parts
TOTAL=$(.venv/bin/python -c "import video_drone as V; print(int(V.DURATION*V.FPS))" 2>/dev/null)
N=8
for i in $(seq 0 $((N-1))); do
  S=$(( TOTAL * i / N )); E=$(( TOTAL * (i+1) / N ))
  CUDA_VISIBLE_DEVICES=$(( 1 + i % 2 )) MUJOCO_GL=egl .venv/bin/python video_drone.py --start $S --end $E --out renders/parts/part_$(printf %02d $i).mp4 > renders/parts/log_$i.txt 2>&1 &
done
wait
ls renders/parts/part_*.mp4 | sed "s/^/file '/; s/$/'/" | sed "s|file 'renders/parts/|file '|" > renders/parts/list.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i renders/parts/list.txt -c copy renders/drone_silent.mp4
ffmpeg -y -loglevel error -i renders/drone_silent.mp4 -f lavfi -i anullsrc=r=48000:cl=stereo -shortest -c:v copy -c:a aac -b:a 128k renders/Drone_Lidar_Nav_LinkedIn.mp4
echo RENDER_DONE
