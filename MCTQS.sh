#!/bin/bash

source /project/bfys/vlipardi/miniconda3/bin/activate /project/bfys/vlipardi/miniconda3/envs/mcts-track
cd /project/bfys/vlipardi/MCTS-TrackReconstruction

python3 -c "from main import run_MCTQS; run_MCTQS($1)"
#python3 -c "from main import postprocess; postprocess($1)"