#!/usr/bin/env bash
# Run a command inside the ADS Python environment.
#   python_utils/adsenv.sh python -m adsutil.smoke_test
#   python_utils/adsenv.sh python my_script.py
export HPEESOF_DIR="${HPEESOF_DIR:-/usr/local/ADS2027}"
export LD_LIBRARY_PATH="$HPEESOF_DIR/lib/linux_x86_64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"

if [ "$1" = python ]; then
    shift
    exec "$HPEESOF_DIR/tools/python/bin/python3" "$@"
fi
exec "$@"
