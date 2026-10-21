#!/bin/bash
# Full LatentSched pipeline for an EC2 run. Stops the instance when finished
# so compute billing ends with the job.
cd ~/LatentSched
source .venv/bin/activate
python -m vae.train_vae
python -m evaluation.evaluate_cluster
for s in random latent-bandit; do
  python -m scheduler.search --target race --strategy $s --trials 200 --seed 0
done
python -m evaluation.evaluate_search --target race
sudo shutdown -h now
