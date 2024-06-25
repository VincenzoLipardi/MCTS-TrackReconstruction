# Define the path to your Python script
python_script="/project/bfys/Xenofon/MCTQS/MCTS-TrackReconstruction/main.py"

for i in {0..100}; do
    qsub -q long -l walltime=48:00:00 -v extra=$i MCTQS.sh
done
