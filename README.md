# Split-Agnostic Meta-learning Evasion (SAME) Attack
To train the MAML algorithm from scratch, we follow the structure of https://github.com/oscarknagg/few-shot/tree/672de83a853cc2d5e9fe304dc100b4a735c10c15
Pretrained models are included and referenced in meta_attack.py. However, probably need to change file path respective to your system. 

To run attack with pre-trained MAML simply run 'python meta_attack.py'
It is currently set up to use the Partial SAME setting of two known data points in support and one in query. Due to time contstraints, it is not as modular as final public repo will be. To change the type of Partial SAME edit lines 150-154, where you can choose how many known data points and if in support or query set. 

If you change the Partial SAME attack setting, you need to change the number of splits to optimize over. Line 196 is the equivalent of "k" in the paper. And then lines 140, 206, 275 should be changed to the maximum number of possible splits. These numbers can be found in the table in the paper. 
