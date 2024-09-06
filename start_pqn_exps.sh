# breakout
sbatch train_real cleanba/cleanba_pqn.py --track
sbatch train_real cleanba/cleanba_impala.py --track
sbatch train_real cleanba/cleanba_ppo.py --track

# pong
sbatch train_real cleanba/cleanba_pqn.py --env_id=Pong-v5 --track
sbatch train_real cleanba/cleanba_impala.py --env_id=Pong-v5 --track
sbatch train_real cleanba/cleanba_ppo.py --env_id=Pong-v5 --track

# pacman
sbatch train_real cleanba/cleanba_pqn.py --env_id=MsPacman-v5 --track
sbatch train_real cleanba/cleanba_impala.py --env_id=MsPacman-v5 --track
sbatch train_real cleanba/cleanba_ppo.py --env_id=MsPacman-v5 --track

# SpaceInvaders
sbatch train_real cleanba/cleanba_pqn.py --env_id=SpaceInvaders-v5 --track
sbatch train_real cleanba/cleanba_impala.py --env_id=SpaceInvaders-v5 --track
sbatch train_real cleanba/cleanba_ppo.py --env_id=SpaceInvaders-v5 --track

