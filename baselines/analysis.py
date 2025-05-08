import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# grid first

bc_grid_single_file_path = "BC/grid_accuracy_BC_2hyp.csv"
bc_grid_group_file_path = "BC/grid_accuracy_BC_2hyp_group.csv"

bc_grid_single_df = pd.read_csv(bc_grid_single_file_path)
bc_grid_group_df = pd.read_csv(bc_grid_group_file_path)

bc_acc_single = bc_grid_single_df["accuracy"].mean()
bc_acc_group = bc_grid_group_df["accuracy"].mean()

print(f"BC accuracy (single): {bc_acc_single}")
print(f"BC accuracy (group): {bc_acc_group}")
print('--------------------------------')

autoToM_single_file_path = "AutoToM/grid_accuracy_AutoToM_2hyp.csv"
autoToM_group_file_path = "AutoToM/grid_accuracy_AutoToM_2hyp_group.csv"

autoToM_single_df = pd.read_csv(autoToM_single_file_path)
autoToM_group_df = pd.read_csv(autoToM_group_file_path)

autoToM_acc_single = autoToM_single_df["accuracy"].mean()
autoToM_acc_group = autoToM_group_df["accuracy"].mean()

print(f"AutoToM accuracy (single): {autoToM_acc_single}")
print(f"AutoToM accuracy (group): {autoToM_acc_group}")
print('--------------------------------')

nllm_single_file_path = "NLLM/grid_accuracy_NLLM_2hyp.csv"
nllm_group_file_path = "NLLM/grid_accuracy_NLLM_2hyp_group.csv"

nllm_single_df = pd.read_csv(nllm_single_file_path)
nllm_group_df = pd.read_csv(nllm_group_file_path)

nllm_acc_single = nllm_single_df["accuracy"].mean()
nllm_acc_group = nllm_group_df["accuracy"].mean()

print(f"NLLM accuracy (single): {nllm_acc_single}")
print(f"NLLM accuracy (group): {nllm_acc_group}")
print('--------------------------------')

# Create data structures to store results for plotting
models = ['BC', 'AutoToM', 'Naive LLM']
single_accs = [bc_acc_single, autoToM_acc_single, nllm_acc_single]
group_accs = [bc_acc_group, autoToM_acc_group, nllm_acc_group]

# Define colors for each model
model_colors = {'BC': 'red', 'AutoToM': 'blue', 'Naive LLM': 'purple', 'FSM': 'green'}

# Store FSM results by number of hypotheses
fsm_hyp_nums = []
fsm_single_accs = []
fsm_group_accs = []

# load the many bootstrap files
bootstrap_hyp_nums = []
bootstrap_accs = []

# Create dictionaries to store data for each configuration
bootstrap_data = {}

for two_stage in [True, False]:
    for structured in ["False", "p1", "p2"]:
        for rejuvenation in [True, False]:
            # Create a configuration key
            config_key = f"{'2stage' if two_stage else 'nostage'}_{'str'+structured if structured != 'False' else 'nostr'}_{'rejuv' if rejuvenation else 'norejuv'}"
            
            # Initialize dictionary for this configuration
            bootstrap_data[config_key] = {}
            
            for top_k in [0, 1, 3, 5, 10, 20]:
                # Rename top_k=0 to top_k=25 for display purposes
                display_top_k = 25 if top_k == 0 else top_k
                
                two_stage_str = "_two_stage" if two_stage else ""
                structured_str = f"_structured_{structured}" if structured != "False" else ""
                top_k_str = f"_topk{top_k}"
                rejuvenation_str = "_rejuvenation" if rejuvenation else ""

                filepath = f"FSM/new_bootstrap_accuracy_FSM{two_stage_str}{structured_str}{rejuvenation_str}{top_k_str}.csv"
                
                try:
                    df = pd.read_csv(filepath)
                    
                    # Group by num_hypothesis and calculate mean accuracy
                    grouped = df.groupby('num_hypothesis')['accuracy'].mean().reset_index()
                    
                    # Store the data for this top_k value
                    bootstrap_data[config_key][display_top_k] = {
                        'hyp_nums': grouped['num_hypothesis'].tolist(),
                        'accuracies': grouped['accuracy'].tolist()
                    }
                    
                    # Store the data for the first plot (to be used later)
                    if two_stage == False and structured == "False" and rejuvenation == False and top_k == 0:
                        bootstrap_hyp_nums = grouped['num_hypothesis'].tolist()
                        bootstrap_accs = grouped['accuracy'].tolist()
                    
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    continue

# Create plots for each configuration with all top_k values on the same plot
for config_key, config_data in bootstrap_data.items():
    if not config_data:  # Skip if no data for this configuration
        continue
    
    plt.figure(figsize=(10, 6))
    
    # Plot baseline models as horizontal lines
    for i, model in enumerate(models):
        plt.axhline(y=single_accs[i], linestyle='--', 
                   label=f'{model}', color=model_colors[model])
    
    # Define a colormap for different top_k values
    top_k_colors = plt.cm.viridis(np.linspace(0, 1, len(config_data)))
    
    # Plot each top_k value with a different color
    for i, (top_k, data) in enumerate(sorted(config_data.items())):
        plt.plot(data['hyp_nums'], data['accuracies'], 
                 marker='o', linestyle='-', 
                 label=f'top_k={top_k}', 
                 color=top_k_colors[i])
    
    plt.xlabel('Number of Hypotheses')
    plt.ylabel('Accuracy')
    plt.title(f'Bootstrap Accuracy: {config_key}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'bootstrap_accuracy_{config_key}.png', dpi=300)
    plt.close()

# Remove the breakpoint() that was in the original code
# breakpoint()

for hypothesis_num in range(1, 13):
    # fsm_single_file_path = f"FSM/grid_accuracy_FSM_{hypothesis_num}hyp.csv"
    fsm_group_file_path = f"FSM/grid_accuracy_FSM_{hypothesis_num}hyp_group.csv"

    # fsm_single_df = pd.read_csv(fsm_single_file_path)
    fsm_group_df = pd.read_csv(fsm_group_file_path)
    
    # Filter to only include rows with 'deepseek' in the 'llm_model' column
    # fsm_single_df = fsm_single_df[fsm_single_df['llm_model'].str.contains('deepseek', case=False, na=False)]
    fsm_group_df = fsm_group_df[fsm_group_df['llm_model'].str.contains('deepseek', case=False, na=False)]

    # fsm_acc_single = fsm_single_df["accuracy"].mean()
    try:
        fsm_acc_group = fsm_group_df["accuracy"].mean()
    except:
        breakpoint()
    
    fsm_hyp_nums.append(hypothesis_num)
    # fsm_single_accs.append(fsm_acc_single)
    fsm_group_accs.append(fsm_acc_group)

    # print(f"FSM {hypothesis_num} accuracy (single): {fsm_acc_single}")
    # print(f"FSM {hypothesis_num} accuracy (group): {fsm_acc_group}")
    # print('--------------------------------')

# Create the side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Single accuracy
for i, model in enumerate(models):
    ax1.axhline(y=single_accs[i], linestyle='--', label=f'{model}', color=model_colors[model])

ax1.set_xlabel('Number of Hypotheses')
ax1.set_ylabel('Accuracy')
ax1.set_title('Single Accuracy')
ax1.set_xticks(bootstrap_hyp_nums)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Group accuracy
for i, model in enumerate(models):
    ax2.axhline(y=group_accs[i], linestyle='--', label=f'{model}', color=model_colors[model])

# Add FSM scatter and line plot
ax2.plot(fsm_hyp_nums, fsm_group_accs, marker='o', linestyle='-', label='FSM', color=model_colors['FSM'])
ax2.scatter(fsm_hyp_nums, fsm_group_accs, color=model_colors['FSM'])

ax2.set_xlabel('Number of Hypotheses')
ax2.set_ylabel('Accuracy')
ax2.set_title('Group Accuracy')
ax2.set_xticks(fsm_hyp_nums)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300)
plt.close()



# human data

bc_human_file_path = "BC/human_accuracy_BC_4hyp.csv"
bc_human_df = pd.read_csv(bc_human_file_path)

bc_human_acc = bc_human_df["accuracy"].mean()

print(f"BC human accuracy: {bc_human_acc}")
print('--------------------------------')

autoToM_human_file_path = "AutoToM/human_accuracy_AutoToM_2hyp.csv"
autoToM_human_df = pd.read_csv(autoToM_human_file_path)

autoToM_human_acc = autoToM_human_df["accuracy"].mean()

print(f"AutoToM human accuracy: {autoToM_human_acc}")
print('--------------------------------')

nllm_human_file_path = "NLLM/human_accuracy_NLLM_2hyp.csv"
nllm_human_df = pd.read_csv(nllm_human_file_path)

nllm_human_acc = nllm_human_df["accuracy"].mean()

print(f"NLLM human accuracy: {nllm_human_acc}")
print('--------------------------------')

for hypothesis_num in range(1, 13):
    fsm_human_file_path = f"FSM/human_accuracy_FSM_{hypothesis_num}hyp.csv"
    fsm_human_df = pd.read_csv(fsm_human_file_path)
    
    # Filter to only include rows with 'deepseek' in the 'llm_model' column
    fsm_human_df = fsm_human_df[fsm_human_df['llm_model'].str.contains('deepseek', case=False, na=False)]

    fsm_human_acc = fsm_human_df["accuracy"].mean()

    print(f"FSM {hypothesis_num} human accuracy: {fsm_human_acc}")
    print('--------------------------------')

# Create data structures to store human results for plotting
human_models = ['BC', 'AutoToM', 'Naive LLM']
human_accs = [bc_human_acc, autoToM_human_acc, nllm_human_acc]

# Store FSM human results by number of hypotheses
fsm_human_hyp_nums = []
fsm_human_accs = []

for hypothesis_num in range(1, 13):
    fsm_human_file_path = f"FSM/human_accuracy_FSM_{hypothesis_num}hyp.csv"
    fsm_human_df = pd.read_csv(fsm_human_file_path)
    
    # Filter to only include rows with 'deepseek' in the 'llm_model' column
    fsm_human_df = fsm_human_df[fsm_human_df['llm_model'].str.contains('deepseek', case=False, na=False)]
    
    fsm_human_acc = fsm_human_df["accuracy"].mean()
    
    fsm_human_hyp_nums.append(hypothesis_num)
    fsm_human_accs.append(fsm_human_acc)

# Create the human accuracy plot
plt.figure(figsize=(10, 6))

# Define colors for each model (same as before for consistency)
model_colors = {'BC': 'red', 'AutoToM': 'blue', 'Naive LLM': 'purple', 'FSM': 'green'}

# Plot horizontal lines for BC, AutoToM, and NLLM
for i, model in enumerate(human_models):
    plt.axhline(y=human_accs[i], linestyle='--', label=f'{model}', color=model_colors[model])

# Add FSM scatter and line plot
plt.plot(fsm_human_hyp_nums, fsm_human_accs, marker='o', linestyle='-', label='FSM', color=model_colors['FSM'])
plt.scatter(fsm_human_hyp_nums, fsm_human_accs, color=model_colors['FSM'])

plt.xlabel('Number of Hypotheses')
plt.ylabel('Accuracy')
plt.title('Human Accuracy Across Models')
plt.xticks(fsm_human_hyp_nums)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('human_accuracy_comparison.png', dpi=300)
plt.close()


# partnr data

autoToM_partnr_file_path = "AutoToM/partnr_accuracy_AutoToM_2hyp.csv"
autoToM_partnr_df = pd.read_csv(autoToM_partnr_file_path)

autoToM_partnr_acc = autoToM_partnr_df["accuracy"].mean()

print(f"AutoToM partnr accuracy: {autoToM_partnr_acc}")
print('--------------------------------')

nllm_partnr_file_path = "NLLM/partnr_accuracy_NLLM_2hyp.csv"
nllm_partnr_df = pd.read_csv(nllm_partnr_file_path)

nllm_partnr_acc = nllm_partnr_df["accuracy"].mean()

print(f"NLLM partnr accuracy: {nllm_partnr_acc}")
print('--------------------------------')

# Store FSM partnr results by number of hypotheses
fsm_partnr_hyp_nums = []
fsm_partnr_accs = []

for hypothesis_num in range(1, 13):
    fsm_partnr_file_path = f"FSM/partnr_accuracy_FSM_{hypothesis_num}hyp.csv"
    fsm_partnr_df = pd.read_csv(fsm_partnr_file_path)
    
    # Filter to only include rows with 'deepseek' in the 'llm_model' column
    fsm_partnr_df = fsm_partnr_df[fsm_partnr_df['llm_model'].str.contains('deepseek', case=False, na=False)]
    
    fsm_partnr_acc = fsm_partnr_df["accuracy"].mean()
    
    fsm_partnr_hyp_nums.append(hypothesis_num)
    fsm_partnr_accs.append(fsm_partnr_acc)

    print(f"FSM {hypothesis_num} partnr accuracy: {fsm_partnr_acc}")
    print('--------------------------------')

# Create the partnr accuracy plot
plt.figure(figsize=(10, 6))

# Define colors for each model (same as before for consistency)
model_colors = {'BC': 'red', 'AutoToM': 'blue', 'Naive LLM': 'purple', 'FSM': 'green'}

# Plot horizontal lines for AutoToM and NLLM
plt.axhline(y=autoToM_partnr_acc, linestyle='--', label='AutoToM', color=model_colors['AutoToM'])
plt.axhline(y=nllm_partnr_acc, linestyle='--', label='Naive LLM', color=model_colors['Naive LLM'])

# Add FSM scatter and line plot
plt.plot(fsm_partnr_hyp_nums, fsm_partnr_accs, marker='o', linestyle='-', label='FSM', color=model_colors['FSM'])
plt.scatter(fsm_partnr_hyp_nums, fsm_partnr_accs, color=model_colors['FSM'])

plt.xlabel('Number of Hypotheses')
plt.ylabel('Accuracy')
plt.title('Partnr Accuracy Across Models')
plt.xticks(fsm_partnr_hyp_nums)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('partnr_accuracy_comparison.png', dpi=300)
plt.close()

# Add ablation plots for bootstrap data components
plt.figure(figsize=(12, 6))

# Extract the best accuracy for each configuration
best_configs = {}

# Process data for ablation plots
for config_key, config_data in bootstrap_data.items():
    if not config_data:  # Skip if no data for this configuration
        continue
    
    # Find the best accuracy across all top_k values and hypothesis numbers
    best_acc = 0
    for top_k, data in config_data.items():
        max_acc = max(data['accuracies']) if data['accuracies'] else 0
        if max_acc > best_acc:
            best_acc = max_acc
    
    best_configs[config_key] = best_acc

# 1. Two-stage ablation plot
plt.figure(figsize=(12, 6))

# Plot baselines first
for i, model in enumerate(models):
    plt.bar(i, single_accs[i], color=model_colors[model], label=model)

# Group configurations by two_stage setting
two_stage_results = {'True': [], 'False': []}
for config_key, acc in best_configs.items():
    if 'nostage' in config_key:
        two_stage_results['False'].append(acc)
    else:
        two_stage_results['True'].append(acc)

# Calculate means
two_stage_means = {k: np.mean(v) if v else 0 for k, v in two_stage_results.items()}

# Plot FSM results
plt.bar(len(models), two_stage_means['False'], color='lightgreen', label='FSM (No Two-Stage)')
plt.bar(len(models) + 1, two_stage_means['True'], color='darkgreen', label='FSM (Two-Stage)')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Effect of Two-Stage Processing')
plt.xticks(range(len(models) + 2), models + ['FSM\n(No Two-Stage)', 'FSM\n(Two-Stage)'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('ablation_two_stage.png', dpi=300)
plt.close()

# 2. Structure ablation plot
plt.figure(figsize=(12, 6))

# Plot baselines first
for i, model in enumerate(models):
    plt.bar(i, single_accs[i], color=model_colors[model], label=model)

# Group configurations by structure setting
structure_results = {'False': [], 'p1': [], 'p2': []}
for config_key, acc in best_configs.items():
    if 'nostr' in config_key:
        structure_results['False'].append(acc)
    elif 'strp1' in config_key:
        structure_results['p1'].append(acc)
    elif 'strp2' in config_key:
        structure_results['p2'].append(acc)

# Calculate means
structure_means = {k: np.mean(v) if v else 0 for k, v in structure_results.items()}

# Plot FSM results
plt.bar(len(models), structure_means['False'], color='lightgreen', label='FSM (No Structure)')
plt.bar(len(models) + 1, structure_means['p1'], color='green', label='FSM (Structure p1)')
plt.bar(len(models) + 2, structure_means['p2'], color='darkgreen', label='FSM (Structure p2)')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Effect of Structure Type')
plt.xticks(range(len(models) + 3), models + ['FSM\n(No Structure)', 'FSM\n(Structure p1)', 'FSM\n(Structure p2)'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('ablation_structure.png', dpi=300)
plt.close()

# 3. Rejuvenation ablation plot
plt.figure(figsize=(12, 6))

# Plot baselines first
for i, model in enumerate(models):
    plt.bar(i, single_accs[i], color=model_colors[model], label=model)

# Group configurations by rejuvenation setting
rejuv_results = {'True': [], 'False': []}
for config_key, acc in best_configs.items():
    if 'norejuv' in config_key:
        rejuv_results['False'].append(acc)
    else:
        rejuv_results['True'].append(acc)

# Calculate means
rejuv_means = {k: np.mean(v) if v else 0 for k, v in rejuv_results.items()}

# Plot FSM results
plt.bar(len(models), rejuv_means['False'], color='lightgreen', label='FSM (No Rejuvenation)')
plt.bar(len(models) + 1, rejuv_means['True'], color='darkgreen', label='FSM (Rejuvenation)')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Effect of Rejuvenation')
plt.xticks(range(len(models) + 2), models + ['FSM\n(No Rejuvenation)', 'FSM\n(Rejuvenation)'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('ablation_rejuvenation.png', dpi=300)
plt.close()

# 4. Particles (top_k) effect plot
plt.figure(figsize=(12, 6))

# Plot baselines as horizontal lines
for i, model in enumerate(models):
    plt.axhline(y=single_accs[i], linestyle='--', label=model, color=model_colors[model])

# Find a representative configuration to show top_k effect
# Using the configuration with the best overall performance
best_config_key = max(best_configs, key=best_configs.get)
print(f"Best configuration: {best_config_key} with accuracy {best_configs[best_config_key]}")

# If best config has data for all top_k values, use it
# Otherwise, find a config with complete data
config_for_topk = best_config_key
if len(bootstrap_data[best_config_key]) < 6:  # Should have 6 top_k values
    for config_key, config_data in bootstrap_data.items():
        if len(config_data) == 6:
            config_for_topk = config_key
            break

# Extract top_k values and their best accuracies
topk_values = []
best_topk_accs = []

for top_k, data in sorted(bootstrap_data[config_for_topk].items()):
    topk_values.append(top_k)
    best_topk_accs.append(max(data['accuracies']))

# Plot the effect of top_k
plt.plot(topk_values, best_topk_accs, marker='o', linestyle='-', 
         label=f'FSM ({config_for_topk})', color='green')

plt.xlabel('Number of Particles (top_k)')
plt.ylabel('Best Accuracy')
plt.title('Effect of Number of Particles')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('ablation_particles.png', dpi=300)
plt.close()

# Find the absolute best configuration
best_config = max(best_configs, key=best_configs.get)
best_accuracy = best_configs[best_config]

# Parse the best configuration
two_stage = "Two-Stage" if "2stage" in best_config else "No Two-Stage"
structured = "No Structure" if "nostr" in best_config else ("Structure p1" if "strp1" in best_config else "Structure p2")
rejuvenation = "Rejuvenation" if "rejuv" in best_config and "norejuv" not in best_config else "No Rejuvenation"

# Find the best top_k for this configuration
best_topk = 0
best_topk_acc = 0
for top_k, data in bootstrap_data[best_config].items():
    max_acc = max(data['accuracies'])
    if max_acc > best_topk_acc:
        best_topk_acc = max_acc
        best_topk = top_k

# Find the best number of hypotheses
best_hyp_num = 0
for top_k, data in bootstrap_data[best_config].items():
    if top_k == best_topk:
        best_hyp_idx = np.argmax(data['accuracies'])
        best_hyp_num = data['hyp_nums'][best_hyp_idx]

print("\nBest Configuration Analysis:")
print(f"Configuration: {best_config}")
print(f"Settings: {two_stage}, {structured}, {rejuvenation}")
print(f"Best top_k: {best_topk}")
print(f"Best number of hypotheses: {best_hyp_num}")
print(f"Best accuracy: {best_accuracy:.4f}")
print(f"Improvement over best baseline ({max(models, key=lambda m: single_accs[models.index(m)])}): {best_accuracy - max(single_accs):.4f}")

# Create a summary plot showing the best configuration vs baselines
plt.figure(figsize=(10, 6))

# Plot baselines
for i, model in enumerate(models):
    plt.bar(i, single_accs[i], color=model_colors[model], label=model)

# Plot best FSM configuration
plt.bar(len(models), best_accuracy, color='green', label=f'Best FSM\n({best_config})')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Best Configuration vs Baselines')
plt.xticks(range(len(models) + 1), models + [f'Best FSM\n{two_stage}\n{structured}\n{rejuvenation}\ntop_k={best_topk}'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('best_configuration.png', dpi=300)
plt.close()

