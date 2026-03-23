#!/bin/bash
# Start virtual display + VNC + noVNC

# Start Xvfb (virtual display)
Xvfb :1 -screen 0 1920x1080x24 &
sleep 1

# Start window manager
DISPLAY=:1 openbox &

# Start x11vnc
x11vnc -display :1 -forever -nopw -quiet &

# Start noVNC (web-based VNC viewer)
websockify --web /usr/share/novnc 6080 localhost:5900 &

echo "Agent VM ready:"
echo "  VNC:   localhost:5900"
echo "  noVNC: http://localhost:6080"

# Keep container alive
wait
