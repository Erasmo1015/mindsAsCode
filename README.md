# Modeling Others' Minds as Code

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

## Setting up baselines

The only baseline that needs additional setup is AutoToM. To have it support more efficient model loading for any huggingface model, we created a custom fork. To install it, do the following:

```
git clone https://github.com/KJha02/AutoToM.git baselines/AutoToM
```

Make sure to log in to your huggingface account and follow instructions on their website to get started.

## Generating ground truth agent trajectories

You can generate the full dataset of ground truth scripted agent behaviors by running 

```
python gen_data.py
```


## Designing agents by hand

The folder `generated_outputs/hand_designed` is where I have .txt files of different agent types.

I run them in the file gen_data.py, which will automatically save a gif of the selected agent.

The agent codes in the hand_designed folder are loaded and sorted alphabetically. You need to specify which agent you want to run by running 

```
python gen_data.py <AGENT_ID_HERE>
```

It defaults to loading the first agent sorted alphabetically just for rendering gifs.


## Evaluating Agents

The most basic evaluation on the gridworld tasks can be performed by running the command

```
python plot_and_eval.py --bootstrap
```

This will run and save results from evaluations of ROTE, bootstrapping the accuracy at few number of hypotheses. **Make sure anytime you evaluate ROTE the "bootstrap" flag is included**. 

You can check the ```plot_and_eval.py``` file for additional arguments and configurations, as well as different baselines to run.

The same file can be used to evaluate models on human gameplay data. To run these eval, do the following:

```
python plot_and_eval.py --human_data True --flip_quarter False --num_steps_to_predict 10
```

Instructions on running the human gameplay and prediction experiments are below.

## Human experiments

First, make sure you have [NiceWebRL](https://github.com/KempnerInstitute/nicewebrl) installed.

Then, generate necessary scripted and tutorial videos using the command

```
python video_generation_script.py
```

To collect gameplay of humans, run

```
python play_human_web_app.py
```

To collect human predictions of AI gameplay, run

```
python prediction_ai_web_app.py
```

To collect human predictions of human gameplay, run

```
python prediction_human_web_app.py
```


## Partnr Evaluations

Clone our fork of the [Partnr](https://github.com/facebookresearch/partnr-planner) repository. Our fork can be found [here: https://github.com/KJha02/partnr-planner](https://github.com/KJha02/partnr-planner).

Follow the instructions on that repository for how to set it up.

The only changes we made we modifying the data saving functionalities and scripts to support the format used in our evals. 

To generate data, follow the commands in the file ```scripts/gen_data.slurm```. Then, you can finetune BC models using the file ```scripts/finetune_planner.sh```. 

To run evals on this dataset after generating data, in this repository you can simply run 

```
python eval_partnr.py
```

again using the variety of different arguments we have provided to change the model you are evaluating.
