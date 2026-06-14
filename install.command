#!/bin/bash
# ================================================================
#  polyfit — macOS installer
#
#  Double-click this file in Finder to install polyfit.
#  A Terminal window will open and guide you through the process.
# ================================================================

# Navigate to the folder containing this script, regardless of
# where Finder or the user launched it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colours (disabled on non-TTY, e.g. redirected output)
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
echo "  2. Create polyfit.app in ~/Applications"
echo ""
echo "You can then launch polyfit by double-clicking"
echo "polyfit.app in your Applications folder."
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
    echo "Please install Python 3 from:  https://www.python.org/downloads/"
    echo "Then double-click this installer again."
    echo ""
    read -rp "Press Enter to close…"
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
    echo "  → Open Finder → Applications → double-click polyfit"
    echo ""
    echo "${YELLOW}Note:${NC} macOS may show a security warning the first time."
    echo "If so: System Settings → Privacy & Security → Open Anyway"
else
    echo "${RED}Installation failed (exit code: $EXIT_CODE).${NC}"
    echo ""
    echo "Please check the error messages above."
    echo "If the problem persists, try running manually:"
    echo ""
    echo "  cd '${SCRIPT_DIR}'"
    echo "  python3 create_app.py"
fi

echo ""
read -rp "Press Enter to close this window…"
osascript -e 'tell application "Terminal" to close front window' &>/dev/null &
