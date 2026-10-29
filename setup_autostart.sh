#!/bin/bash
# setup_autostart.sh
# Creates an autostart desktop entry for the Jetson Orin.

AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/ai-sewing.desktop"
PROJECT_DIR="$(pwd)"

mkdir -p "$AUTOSTART_DIR"

cat << EOF > "$DESKTOP_FILE"
[Desktop Entry]
Type=Application
Exec=python3 "$PROJECT_DIR/gui_main.py"
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name[en_US]=AI Sewing GUI
Name=AI Sewing GUI
Comment=Start AI Sewing GUI on boot
EOF

echo "Autostart configured at $DESKTOP_FILE"
chmod +x "$PROJECT_DIR/gui_main.py"
echo "You can now reboot to test the autostart or run ./gui_main.py directly."
