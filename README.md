# Split-Agnostic Meta-learning Evasion (SAME) Attack

Code to reproduce our paper "Full and Partial Split Agnositc Meta-Learning Evasion Attacks" will post link once by conference. 

## Setup
### Requirements
Listed in requirements.txt. Install with pip install -r requirements.txt preferably in a virtualenv.

### Data
Edit the DATA_PATH variable in config.py to the location where you store the Omniglot and miniImagenet datasets.

### MAML Training
Following (https://github.com/oscarknagg/few-shot/blob/672de83a853cc2d5e9fe304dc100b4a735c10c15/README.md) 
Run experiments/maml.py to reproduce results from Model-Agnostic Meta-Learning (Finn et al). 
#### Arguments
To reproduce the results in the paper train the MAML with the following command: 'python -m experiments.maml --dataset miniImageNet --order 1 --n 5 --k 2 --q 5 --meta-batch-size 4 --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 5 --epoch-len 25' 

### Meta Attack
Once trained, to run the SAME attack on the MAML and test the results, simply run 'python meta_attack.py'



