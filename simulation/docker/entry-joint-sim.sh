#!/bin/sh
export PYTHONPATH=/build/simulation
export MUJOCO_GL=osmesa
exec /opt/venv/bin/python -c "import sys; from bridge.joint_sim_shim import main; sys.exit(main(sys.argv))" "$@"
