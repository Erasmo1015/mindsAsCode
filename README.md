# From Actions to Automata: Modeling others' minds as code

## Setting up the environment

First, setup a conda environment with Python 3.12

Then, make sure you have the following packages installed

```
flax==0.10.3
jax==0.5.0
jaxlib==0.5.0
numpy
matplotlib
imageio
vllm
transformers
```

Make sure to log in to your hugginface account and follow instructions on their website to get started.

## Designing agents by hand

The folder `generated_outputs/hand_designed' is where I have .txt files of different agent types. 

I run them in the file gen_data.py, which will automatically save a gif of the selected agent.

The agent codes in the hand_designed folder are loaded and sorted alphabetically. You need to specify which agent you want to run by running 

```
python gen_data.py <AGENT_ID_HERE>
```

It defaults to loading the first agent sorted alphabetically.