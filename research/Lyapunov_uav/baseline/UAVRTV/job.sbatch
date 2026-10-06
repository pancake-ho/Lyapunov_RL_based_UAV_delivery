#!/bin/bash
#SBATCH --signal=B:USR1@120
set -euo pipefail
exec "$1" baseline/UAVRTV/main.py "$2" --settings "$3"
