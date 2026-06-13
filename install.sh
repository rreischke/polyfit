#!/bin/bash
# ================================================================
#  polyfit — Linux installer
#
#  Run with:  bash install.sh
#  Or make executable and double-click in a file manager that
#  supports running shell scripts (Nautilus: Edit → Preferences →
#  Behaviour → "Run executable text files when they are opened").
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colours (disabled on non-TTY)
if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'; BOLD=$'\033[1m'; NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

clear
echo "${BOLD}========================================"
echo "         polyfit — Installer"
echo "========================================"
echo "${NC}"
echo "This installer will:"
echo "  1. Install the polyfit Python package"
echo "  2. Add a 'polyfit' command to ~/.local/bin"
echo "  3. Add a polyfit entry to your application launcher"
echo ""
echo "------------------------------------------------"
echo ""

# ----------------------------------------------------------------
# Check Python 3
# ----------------------------------------------------------------
echo "Checking for Python 3…"

PY=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        if [ "$MAJOR" = "3" ]; then
            PY="$cmd"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo ""
    echo "${RED}ERROR: Python 3 was not found on this system.${NC}"
    echo ""
    echo "Install it with:"
    echo "  sudo apt install python3 python3-pip     # Debian / Ubuntu"
    echo "  sudo dnf install python3 python3-pip     # Fedora / RHEL"
    echo "  sudo pacman -S python python-pip          # Arch"
    echo ""
    echo "Then run this installer again."
    exit 1
fi

echo "${GREEN}✓  Found: $($PY --version)${NC}"
echo ""

# ----------------------------------------------------------------
# Run the Python installer
# ----------------------------------------------------------------
echo "Starting installation…"
echo ""

"$PY" "$SCRIPT_DIR/create_app.py"
EXIT_CODE=$?
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "${GREEN}${BOLD}Installation complete!${NC}"
    echo ""
    echo "  Launch from terminal:  polyfit"
    echo "  Or search for 'polyfit' in your application launcher."
    echo ""
    echo "${YELLOW}Note:${NC} If the 'polyfit' command is not found, add ~/.local/bin to PATH:"
    echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
else
    echo "${RED}Installation failed (exit code: $EXIT_CODE).${NC}"
    echo ""
    echo "Please check the error messages above."
    echo "If the problem persists, try running manually:"
    echo ""
    echo "  cd '${SCRIPT_DIR}'"
    echo "  python3 create_app.py"
fi
