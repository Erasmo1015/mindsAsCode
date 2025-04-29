import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


bc_file = "BC/accuracy_BC_4hyp.csv"
bc_df = pd.read_csv(bc_file)
bc_df['num_hypothesis'] = 4

fsm_files = []
for hyp in [2,3,4,5]:
    fsm_files.append(f"FSM/accuracy_FSM_{hyp}hyp.csv")

autoToM_file = "AutoToM/accuracy_AutoToM_2hyp.csv"
autoToM_df = pd.read_csv(autoToM_file)
autoToM_df['num_hypothesis'] = 2

fsm_dfs = []
for fsm_file, hyp in zip(fsm_files, [2,3,4,5]):
    df = pd.read_csv(fsm_file)
    df['num_hypothesis'] = hyp
    df['approach'] = 'FSM'  # Add approach label
    fsm_dfs.append(df)

fsm_df = pd.concat(fsm_dfs, ignore_index=True)

# Add approach label to autoToM_df
autoToM_df['approach'] = 'AutoToM'

# Add approach label to bc_df
bc_df['approach'] = 'MToM'

all_df = pd.concat([bc_df, fsm_df, autoToM_df], ignore_index=True)

sns.set_context("paper")

# Plot BC baseline as horizontal line
bc_accuracy = bc_df['accuracy'].iloc[0]
plt.axhline(y=bc_accuracy, color='red', linestyle='--', label='MToM')

# Plot horizontal lines for each LLM model in autoToM_df
autoToM_models = autoToM_df['llm_model'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(autoToM_models)))
for i, model in enumerate(autoToM_models):
    model_accuracy = autoToM_df[autoToM_df['llm_model'] == model]['accuracy'].iloc[0]
    plt.axhline(y=model_accuracy, color=colors[i], linestyle=':', 
                label=f'AutoToM ({model})')

# Plot FSM line + scatter with approach in label
sns.lineplot(data=fsm_df, x='num_hypothesis', y='accuracy', hue='llm_model', 
             style='llm_model', hue_norm=None)
sns.scatterplot(data=fsm_df, x='num_hypothesis', y='accuracy', hue='llm_model', 
                style='llm_model', hue_norm=None)

# Update legend labels to include approach
handles, labels = plt.gca().get_legend_handles_labels()
for i in range(len(labels)):
    if labels[i] in fsm_df['llm_model'].unique():
        labels[i] = f'FSM ({labels[i]})'

# Position the legend below the plot to avoid overlap
plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           ncol=3, frameon=True)  # Use ncol to make legend more compact horizontally

# Adjust the figure size and layout to accommodate the legend
plt.gcf().set_size_inches(10, 8)  # Increase vertical size to make room for legend
plt.tight_layout()

plt.xlabel('Number of Hypotheses')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs Number of Hypotheses')
plt.savefig('accuracy_vs_num_hypotheses.pdf', dpi=300, bbox_inches='tight')
plt.close()
