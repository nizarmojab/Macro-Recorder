FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:1

# System dependencies
RUN apt-get update && apt-get install -y \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    openbox \
    xterm \
    python3 \
    python3-pip \
    python3-tk \
    scrot \
    xdotool \
    wmctrl \
    libx11-dev \
    libxtst-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages for interaction inside the VM
RUN pip3 install pyautogui pywinauto playwright mss Pillow

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# noVNC setup
RUN mkdir -p /opt/novnc/utils/websockify
RUN ln -s /usr/share/novnc/utils/launch.sh /opt/novnc/launch.sh || true

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5900 6080

ENTRYPOINT ["/entrypoint.sh"]
