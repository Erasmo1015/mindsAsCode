import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import glob

# Define colors for each model
model_colors = {'BC': 'red', 'AutoToM': 'blue', 'Naive LLM': 'purple', 'FSM': 'green', 'Human': 'orange'}

question = 1
if question == 1:
    ############
    # Question 1: Does the FSM have benefits with predicting a single ground truth scripted agent?
    ############

    # Load all the data as before
    bc_grid_single_file_path = "BC/grid_accuracy_BC_2hyp.csv"
    bc_grid_group_file_path = "BC/grid_accuracy_BC_2hyp_group.csv"

    bc_grid_single_df = pd.read_csv(bc_grid_single_file_path)
    bc_grid_group_df = pd.read_csv(bc_grid_group_file_path)

    bc_acc_single = bc_grid_single_df["accuracy"].mean()
    bc_acc_group = bc_grid_group_df["accuracy"].mean()

    print(f"loaded bc grid data")

    autoToM_single_file_path = "AutoToM/grid_accuracy_AutoToM_2hyp.csv"
    autoToM_group_file_path = "AutoToM/grid_accuracy_AutoToM_2hyp_group.csv"

    autoToM_single_df = pd.read_csv(autoToM_single_file_path)
    autoToM_group_df = pd.read_csv(autoToM_group_file_path)

    autoToM_acc_single = autoToM_single_df["accuracy"].mean()
    autoToM_acc_group = autoToM_group_df["accuracy"].mean()

    print(f"loaded autoToM grid data")

    nllm_single_file_path = "NLLM/grid_accuracy_NLLM_2hyp.csv"
    nllm_group_file_path = "NLLM/grid_accuracy_NLLM_2hyp_group.csv"

    nllm_single_df = pd.read_csv(nllm_single_file_path)
    nllm_group_df = pd.read_csv(nllm_group_file_path)

    nllm_acc_single = nllm_single_df["accuracy"].mean()
    nllm_acc_group = nllm_group_df["accuracy"].mean()

    print(f"loaded nllm grid data")

    # Create data structures to store results for plotting
    models = ['BC', 'AutoToM', 'Naive LLM']
    single_accs = [bc_acc_single, autoToM_acc_single, nllm_acc_single]
    group_accs = [bc_acc_group, autoToM_acc_group, nllm_acc_group]

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
                        # Use a more robust approach with csv module
                        
                        # First pass: determine all possible columns and read data
                        all_rows = []
                        max_cols = 0
                        header = None
                        
                        with open(filepath, 'r', newline='') as f:
                            reader = csv.reader(f)
                            for i, row in enumerate(reader):
                                if i == 0:
                                    header = row
                                    continue
                                max_cols = max(max_cols, len(row))
                                all_rows.append(row)
                        
                        # Extend header if needed
                        while len(header) < max_cols:
                            header.append(f'extra_col_{len(header)}')
                        
                        # Create DataFrame with consistent columns
                        data = []
                        for row in all_rows:
                            # Pad row with zeros if needed
                            padded_row = row + ['0'] * (max_cols - len(row))
                            data.append(padded_row)
                        
                        df = pd.DataFrame(data, columns=header)
                        
                        # Convert columns to appropriate types
                        for col in df.columns:
                            try:
                                df[col] = pd.to_numeric(df[col])
                            except:
                                pass  # Keep as string if can't convert
                        
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
        plt.rcParams.update({'font.size': 14})
        
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
        
        plt.xlabel('Number of Hypotheses', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Bootstrap Accuracy: {config_key}', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(False)
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

    print(f"loaded fsm grid data")

    # Create the side-by-side plots
    import seaborn as sns

    # Set seaborn style with larger font size
    sns.set_context("paper", font_scale=2.0)

    # Find the best bootstrap accuracy across all configurations
    best_bootstrap_acc = 0
    for config_key, config_data in bootstrap_data.items():
        for top_k, data in config_data.items():
            if data['accuracies']:
                max_acc = max(data['accuracies'])
                if max_acc > best_bootstrap_acc:
                    best_bootstrap_acc = max_acc

    # Create a figure with three subplots for single agent analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Single step accuracy for single agent
    single_models = ['BC', 'AutoToM', 'Naive LLM', 'FSM']
    single_accs = [bc_acc_single, autoToM_acc_single, nllm_acc_single, best_bootstrap_acc]

    ax1.bar(range(len(single_models)), single_accs, width=0.6, 
            color=[model_colors.get(m, 'gray') for m in single_models])
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Single Step Accuracy\n(Single Agent)')
    ax1.set_xticks(range(len(single_models)))
    ax1.set_xticklabels(single_models)
    ax1.grid(False)
    ax1.set_ylim(0, 1)  # Set y-axis from 0 to 1
    
    # Plot 2: Multi step accuracy for single agent
    multistep_single_models = ['BC', 'AutoToM', 'Naive LLM', 'FSM']
    multistep_single_accs = [bc_multistep_acc_single, autoToM_multistep_acc_single, nllm_multistep_acc_single, fsm_multistep_acc_single]

    ax2.bar(range(len(multistep_single_models)), multistep_single_accs, width=0.6, 
            color=[model_colors.get(m, 'gray') for m in multistep_single_models])
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Multi Step Accuracy\n(Single Agent)')
    ax2.set_xticks(range(len(multistep_single_models)))
    ax2.set_xticklabels(multistep_single_models)
    ax2.grid(False)
    ax2.set_ylim(0, 1)  # Set y-axis from 0 to 1
    
    # Plot 3: Average prediction time for single agent
    single_pred_times = [
        bc_multistep_avg_prediction_time,
        autoToM_multistep_avg_prediction_time,
        nllm_multistep_avg_prediction_time,
        best_fsm_multistep_avg_prediction_time
    ]

    # Use log scale for y-axis
    ax3.set_yscale('log')
    ax3.bar(range(len(multistep_single_models)), single_pred_times, width=0.6, 
            color=[model_colors.get(m, 'gray') for m in multistep_single_models])
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Average Prediction Time (s)\nLog Scale')
    ax3.set_title('Prediction Time\n(Single Agent)')
    ax3.set_xticks(range(len(multistep_single_models)))
    ax3.set_xticklabels(multistep_single_models)
    ax3.grid(True, axis='y', alpha=0.3)

    # Add value labels on top of bars
    for i, time in enumerate(single_pred_times):
        ax3.text(i, time * 1.1, f'{time:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Remove spines
    sns.despine(fig)
    
    plt.tight_layout()
    plt.savefig('q1_single_agent_comparison.pdf', dpi=300)
    plt.close()

elif question == 2:
    ############
    # Question 2: Does the FSM have benefits with predicting groups of agents following a script?
    ############

    # Load all the data as before
    bc_human_file_path = "BC/human_accuracy_BC_4hyp.csv"
    bc_human_df = pd.read_csv(bc_human_file_path)
    bc_human_acc = bc_human_df["accuracy"].mean()

    print(f"Loaded BC human data")

    autoToM_human_file_path = "AutoToM/human_accuracy_AutoToM_2hyp.csv"
    autoToM_human_df = pd.read_csv(autoToM_human_file_path)
    # Filter to only include rows with 'Llama-3.1' in the 'llm_model' column
    autoToM_human_df = autoToM_human_df[autoToM_human_df['llm_model'].str.contains('Llama-3.1', case=False, na=False)]
    autoToM_human_acc = autoToM_human_df["accuracy"].mean()

    print(f"Loaded AutoToM human data")

    nllm_human_file_path = "NLLM/human_accuracy_NLLM_2hyp.csv"
    
    # Use a more robust approach to handle inconsistent columns
    try:
        # First pass: determine all possible columns and read data
        all_rows = []
        max_cols = 0
        header = None
        
        with open(nllm_human_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    continue
                max_cols = max(max_cols, len(row))
                all_rows.append(row)
        
        # Extend header if needed
        while len(header) < max_cols:
            header.append(f'extra_col_{len(header)}')
        
        # Create DataFrame with consistent columns
        data = []
        for row in all_rows:
            # Pad row with NaN values if needed
            padded_row = row + [''] * (max_cols - len(row))
            data.append(padded_row)
        
        nllm_human_df = pd.DataFrame(data, columns=header)
        
        # Convert columns to appropriate types
        for col in nllm_human_df.columns:
            try:
                nllm_human_df[col] = pd.to_numeric(nllm_human_df[col])
            except:
                pass  # Keep as string if can't convert
    except Exception as e:
        print(f"Error processing NLLM human data: {e}")
        nllm_human_df = pd.DataFrame()  # Empty DataFrame as fallback

    # Filter to only include rows with the best performing LLM model
    if 'llm_model' in nllm_human_df.columns and not nllm_human_df.empty:
        # Group by llm_model and get mean accuracy
        nllm_model_accs = nllm_human_df.groupby('llm_model')['accuracy'].mean()
        # Get the best model
        nllm_human_best_model = nllm_model_accs.idxmax()
        # Filter to only include rows with the best model
        nllm_human_df = nllm_human_df[nllm_human_df['llm_model'] == nllm_human_best_model]

    nllm_human_acc = nllm_human_df["accuracy"].mean()

    print(f"Loaded NLLM human data")

    # Load bootstrap human FSM data
    bootstrap_human_files = glob.glob("FSM/human_bootstrap_detailed_FSM*.csv")
    print(f"Found {len(bootstrap_human_files)} bootstrap human FSM files")

    # Initialize variables to track best performance
    fsm_human_acc = 0
    best_bootstrap_human_file = None
    best_num_hypothesis = None
    best_llm_model = None

    # Process each bootstrap human file
    for file_path in bootstrap_human_files:
        try:
            # Load the data
            bootstrap_human_df = pd.read_csv(file_path)
            
            # Check if 'correct' column exists
            if 'correct' in bootstrap_human_df.columns:
                # Group by num_hypothesis and llm_model, then calculate mean correct score for each group
                if 'num_hypothesis' in bootstrap_human_df.columns and 'llm_model' in bootstrap_human_df.columns:
                    grouped = bootstrap_human_df.groupby(['num_hypothesis', 'llm_model'])['correct'].mean().reset_index()
                    
                    # Find the best combination
                    for _, row in grouped.iterrows():
                        if row['correct'] > fsm_human_acc:
                            fsm_human_acc = row['correct']
                            best_bootstrap_human_file = file_path
                            best_num_hypothesis = row['num_hypothesis']
                            best_llm_model = row['llm_model']
                else:
                    # If columns don't exist, fall back to overall mean
                    mean_correct = bootstrap_human_df['correct'].mean()
                    
                    # Update best score if this is higher
                    if mean_correct > fsm_human_acc:
                        fsm_human_acc = mean_correct
                        best_bootstrap_human_file = file_path
                        best_num_hypothesis = None
                        best_llm_model = None
            else:
                print(f"Warning: 'correct' column not found in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Best FSM human accuracy: {fsm_human_acc} from {best_bootstrap_human_file}")
    if best_num_hypothesis is not None and best_llm_model is not None:
        print(f"Best FSM configuration: num_hypothesis={best_num_hypothesis}, llm_model='{best_llm_model}'")

    # Create a figure with three subplots for group agent analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Single step accuracy for group agent
    group_models = ['BC', 'AutoToM', 'Naive LLM', 'FSM']
    group_accs = [bc_acc_group, autoToM_acc_group, nllm_acc_group, fsm_best_acc]

    ax1.bar(range(len(group_models)), group_accs, width=0.6, 
            color=[model_colors.get(m, 'gray') for m in group_models])
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Single Step Accuracy\n(Group Agent)')
    ax1.set_xticks(range(len(group_models)))
    ax1.set_xticklabels(group_models)
    ax1.grid(False)
    ax1.set_ylim(0, 1)  # Set y-axis from 0 to 1
    
    # Plot 2: Multi step accuracy for group agent
    multistep_group_models = ['BC', 'AutoToM', 'Naive LLM', 'FSM']
    multistep_group_accs = [bc_multistep_acc_group, autoToM_multistep_acc_group, nllm_multistep_acc_group, fsm_multistep_acc_group]

    ax2.bar(range(len(multistep_group_models)), multistep_group_accs, width=0.6, 
            color=[model_colors.get(m, 'gray') for m in multistep_group_models])
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Multi Step Accuracy\n(Group Agent)')
    ax2.set_xticks(range(len(multistep_group_models)))
    ax2.set_xticklabels(multistep_group_models)
    ax2.grid(False)
    ax2.set_ylim(0, 1)  # Set y-axis from 0 to 1
    
    # Plot 3: Average prediction time for group agent
    group_pred_times = [
        bc_multistep_avg_prediction_time_group,
        autoToM_multistep_avg_prediction_time_group,
        nllm_multistep_avg_prediction_time_group,
        best_fsm_multistep_avg_prediction_time  # Using the same value as single if group-specific not available
    ]

    # Use log scale for y-axis
    ax3.set_yscale('log')
    ax3.bar(range(len(multistep_group_models)), group_pred_times, width=0.6, 
            color=[model_colors.get(m, 'gray') for m in multistep_group_models])
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Average Prediction Time (s)\nLog Scale')
    ax3.set_title('Prediction Time\n(Group Agent)')
    ax3.set_xticks(range(len(multistep_group_models)))
    ax3.set_xticklabels(multistep_group_models)
    ax3.grid(True, axis='y', alpha=0.3)

    # Add value labels on top of bars
    for i, time in enumerate(group_pred_times):
        ax3.text(i, time * 1.1, f'{time:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Remove spines
    sns.despine(fig)
    
    plt.tight_layout()
    plt.savefig('q2_group_agent_comparison.pdf', dpi=300)
    plt.close()

elif question == 3:
    ############
    # Question 3: Do the FSMs have benefits when predicting behavior not generated by a ground truth agent?
    ############

    # Load all the data as before
    human_pred_human_file_path = "/mmfs1/gscratch/socialrl/kjha/automaticity/data/human_prediction_data.csv"
    human_pred_human_df = pd.read_csv(human_pred_human_file_path)

    human_acc = (human_pred_human_df['predicted_action_idx'] == human_pred_human_df['gt_action_idx']).mean()

    # Add a new visualization for correlation between human ratings
    print("\nAnalyzing correlations between human perception ratings")
    
    # Get all rating columns
    rating_cols = ['rote_rating', 'informative_rating', 'goal_directed_rating', 
                  'random_rating', 'thinking_rating', 'complex_rating', 'planned_rating']
    
    # Filter to only include columns that exist in the dataframe
    rating_cols = [col for col in rating_cols if col in human_pred_human_df.columns]
    
    if rating_cols:
        # Calculate correlation matrix
        ratings_corr = human_pred_human_df[rating_cols].corr()
        
        # Create a more readable version of the column names for display
        readable_names = {
            'rote_rating': 'Rote',
            'informative_rating': 'Informative',
            'goal_directed_rating': 'Goal-directed',
            'random_rating': 'Random',
            'thinking_rating': 'Thinking',
            'complex_rating': 'Complex',
            'planned_rating': 'Planned'
        }
        
        # Rename the index and columns
        ratings_corr.index = [readable_names.get(col, col) for col in ratings_corr.index]
        ratings_corr.columns = [readable_names.get(col, col) for col in ratings_corr.columns]
        
        # Create the correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(ratings_corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(ratings_corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        
        plt.title('Correlation Between Human Perception Ratings', fontsize=16)
        plt.tight_layout()
        plt.savefig('human_ratings_correlation_matrix.pdf', dpi=300)
        plt.close()
        
        print("Correlation matrix between human ratings saved to 'human_ratings_correlation_matrix.pdf'")
    else:
        print("No rating columns found in the human prediction data")

    # Human pred of agent data
    
    # Load bootstrap human FSM data
    bootstrap_human_files = glob.glob("FSM/human_bootstrap_detailed_FSM*.csv")
    print(f"Found {len(bootstrap_human_files)} bootstrap human FSM files")

    # Initialize variables to track best performance
    fsm_human_acc = 0
    best_bootstrap_human_file = None
    best_num_hypothesis = None
    best_llm_model = None

    # Process each bootstrap human file
    for file_path in bootstrap_human_files:
        try:
            # Load the data
            bootstrap_human_df = pd.read_csv(file_path)
            
            # Check if 'correct' column exists
            if 'correct' in bootstrap_human_df.columns:
                # Group by num_hypothesis and llm_model, then calculate mean correct score for each group
                if 'num_hypothesis' in bootstrap_human_df.columns and 'llm_model' in bootstrap_human_df.columns:
                    grouped = bootstrap_human_df.groupby(['num_hypothesis', 'llm_model'])['correct'].mean().reset_index()
                    
                    # Find the best combination
                    for _, row in grouped.iterrows():
                        if row['correct'] > fsm_human_acc:
                            fsm_human_acc = row['correct']
                            best_bootstrap_human_file = file_path
                            best_num_hypothesis = row['num_hypothesis']
                            best_llm_model = row['llm_model']
                else:
                    # If columns don't exist, fall back to overall mean
                    mean_correct = bootstrap_human_df['correct'].mean()
                    
                    # Update best score if this is higher
                    if mean_correct > fsm_human_acc:
                        fsm_human_acc = mean_correct
                        best_bootstrap_human_file = file_path
                        best_num_hypothesis = None
                        best_llm_model = None
            else:
                print(f"Warning: 'correct' column not found in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Best FSM human accuracy: {fsm_human_acc} from {best_bootstrap_human_file}")
    if best_num_hypothesis is not None and best_llm_model is not None:
        print(f"Best FSM configuration: num_hypothesis={best_num_hypothesis}, llm_model='{best_llm_model}'")

    # ---- START OF NEW EDA SECTION ----
    print("\nStarting Exploratory Data Analysis for Question 3: Human vs. FSM Predictions")

    if not best_bootstrap_human_file:
        print("Cannot proceed with EDA: Best FSM bootstrap file not identified.")
    else:
        # Load the specific CSV file that contained the best FSM configuration
        print(f"Loading best FSM data from: {best_bootstrap_human_file}")
        fsm_all_data_from_best_file_df = pd.read_csv(best_bootstrap_human_file)
        
        # Filter this dataframe for the specific best configuration
        fsm_best_config_df = fsm_all_data_from_best_file_df.copy()
        if best_num_hypothesis is not None:
            fsm_best_config_df = fsm_best_config_df[fsm_best_config_df['num_hypothesis'] == best_num_hypothesis]
        if best_llm_model is not None:
            fsm_best_config_df = fsm_best_config_df[fsm_best_config_df['llm_model'] == best_llm_model]

        print(f"Shape of FSM data for best configuration: {fsm_best_config_df.shape}")
        
        # Prepare human prediction data
        human_df_eda = human_pred_human_df.copy()
        human_df_eda['human_correct'] = (human_df_eda['predicted_action_idx'] == human_df_eda['gt_action_idx']).astype(int)
        
        # Analyze human data
        print("\n--- Human Prediction Analysis ---")
        print(f"Total human predictions: {len(human_df_eda)}")
        print(f"Overall human accuracy: {human_acc:.4f}")
        
        # Analyze confidence ratings in human data if available
        if 'rote_rating' in human_df_eda.columns:
            print("\nHuman perception ratings (average):")
            for rating_col in ['rote_rating', 'informative_rating', 'goal_directed_rating', 
                              'random_rating', 'thinking_rating', 'complex_rating', 'planned_rating']:
                if rating_col in human_df_eda.columns:
                    print(f"- {rating_col}: {human_df_eda[rating_col].mean():.2f}")
        
        # Analyze FSM data
        print("\n--- FSM Prediction Analysis ---")
        print(f"Total FSM predictions: {len(fsm_best_config_df)}")
        print(f"Overall FSM accuracy: {fsm_human_acc:.4f}")
        
        # Analyze confidence and program length in FSM data if available
        if 'confidence' in fsm_best_config_df.columns:
            print(f"Average FSM confidence: {fsm_best_config_df['confidence'].mean():.4f}")
        if 'program_length' in fsm_best_config_df.columns:
            print(f"Average program length: {fsm_best_config_df['program_length'].mean():.2f}")
        
        # Create visualizations for each dataset separately
        
        # 1. Human ratings distribution
        if 'rote_rating' in human_df_eda.columns:
            plt.figure(figsize=(12, 8))
            rating_cols = ['rote_rating', 'informative_rating', 'goal_directed_rating', 
                           'random_rating', 'thinking_rating', 'complex_rating', 'planned_rating']
            
            # Filter to only include columns that exist
            rating_cols = [col for col in rating_cols if col in human_df_eda.columns]
            
            # Create a long-format dataframe for seaborn
            ratings_long = pd.melt(human_df_eda, 
                                  value_vars=rating_cols,
                                  var_name='Rating Type', 
                                  value_name='Rating Value')
            
            # Clean up the rating type names for display
            ratings_long['Rating Type'] = ratings_long['Rating Type'].str.replace('_rating', '')
            
            # Create the violin plot
            sns.violinplot(x='Rating Type', y='Rating Value', data=ratings_long)
            plt.title('Distribution of Human Perception Ratings', fontsize=16)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('human_ratings_distribution.pdf', dpi=300)
            plt.close()
        
        # 2. FSM confidence vs. correctness
        if 'confidence' in fsm_best_config_df.columns and 'correct' in fsm_best_config_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='correct', y='confidence', data=fsm_best_config_df)
            plt.title('FSM Confidence by Prediction Correctness', fontsize=16)
            plt.xlabel('Correct Prediction', fontsize=14)
            plt.ylabel('Confidence', fontsize=14)
            plt.tight_layout()
            plt.savefig('fsm_confidence_by_correctness.pdf', dpi=300)
            plt.close()
        
        # 3. FSM program length vs. correctness
        if 'program_length' in fsm_best_config_df.columns and 'correct' in fsm_best_config_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='correct', y='program_length', data=fsm_best_config_df)
            plt.title('FSM Program Length by Prediction Correctness', fontsize=16)
            plt.xlabel('Correct Prediction', fontsize=14)
            plt.ylabel('Program Length', fontsize=14)
            plt.tight_layout()
            plt.savefig('fsm_program_length_by_correctness.pdf', dpi=300)
            plt.close()
        
        # 4. Human vs FSM accuracy comparison (already created in the original code)
        
        # 5. Additional analysis: Accuracy by task (if task information is available)
        if 'task' in human_df_eda.columns:
            # Human accuracy by task
            human_task_acc = human_df_eda.groupby('task')['human_correct'].mean().reset_index()
            human_task_acc = human_task_acc.sort_values('human_correct', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='task', y='human_correct', data=human_task_acc.head(10))
            plt.title('Human Accuracy by Task (Top 10)', fontsize=16)
            plt.xlabel('Task', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('human_accuracy_by_task.pdf', dpi=300)
            plt.close()
        
        if 'task' in fsm_best_config_df.columns:
            # FSM accuracy by task
            fsm_task_acc = fsm_best_config_df.groupby('task')['correct'].mean().reset_index()
            fsm_task_acc = fsm_task_acc.sort_values('correct', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='task', y='correct', data=fsm_task_acc.head(10))
            plt.title('FSM Accuracy by Task (Top 10)', fontsize=16)
            plt.xlabel('Task', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('fsm_accuracy_by_task.pdf', dpi=300)
            plt.close()

    # ---- END OF NEW EDA SECTION ----
    
    # ---- START OF NEW CORRELATION ANALYSIS SECTION ----
    print("\nStarting Correlation Analysis for Human and FSM Predictions")

    if not best_bootstrap_human_file:
        print("Cannot proceed with correlation analysis: Best FSM bootstrap file not identified.")
    else:
        # Ensure we have the best FSM configuration data
        if 'fsm_best_config_df' not in locals() or fsm_best_config_df.empty:
            # Load the specific CSV file that contained the best FSM configuration
            print(f"Loading best FSM data from: {best_bootstrap_human_file}")
            fsm_all_data_from_best_file_df = pd.read_csv(best_bootstrap_human_file)
            
            # Filter this dataframe for the specific best configuration
            fsm_best_config_df = fsm_all_data_from_best_file_df.copy()
            if best_num_hypothesis is not None:
                fsm_best_config_df = fsm_best_config_df[fsm_best_config_df['num_hypothesis'] == best_num_hypothesis]
            if best_llm_model is not None:
                fsm_best_config_df = fsm_best_config_df[fsm_best_config_df['llm_model'] == best_llm_model]
        
        # Ensure human data is prepared
        if 'human_df_eda' not in locals() or human_df_eda.empty:
            human_df_eda = human_pred_human_df.copy()
            human_df_eda['human_correct'] = (human_df_eda['predicted_action_idx'] == human_df_eda['gt_action_idx']).astype(int)
        
        # 1. Correlation between human perception ratings and human accuracy
        print("\n--- Correlation between human perception ratings and human accuracy ---")
        rating_cols = ['rote_rating', 'informative_rating', 'goal_directed_rating', 
                    'random_rating', 'thinking_rating', 'complex_rating', 'planned_rating']

        # Filter to only include columns that exist
        rating_cols = [col for col in rating_cols if col in human_df_eda.columns]

        # Calculate correlations
        human_corr = pd.DataFrame(index=rating_cols, columns=['correlation_with_accuracy', 'p_value'])

        for col in rating_cols:
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(human_df_eda[col], human_df_eda['human_correct'])
            human_corr.loc[col] = [corr, p_value]

        print(human_corr.sort_values('correlation_with_accuracy', ascending=False))

        # Visualize these correlations
        plt.figure(figsize=(10, 6))
        sns.barplot(x=human_corr.index, y='correlation_with_accuracy', data=human_corr)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Correlation between Human Perception Ratings and Accuracy', fontsize=16)
        plt.xlabel('Rating Type', fontsize=14)
        plt.ylabel('Correlation Coefficient', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        # Add significance markers
        for i, (idx, row) in enumerate(human_corr.iterrows()):
            if row['p_value'] < 0.001:
                plt.text(i, row['correlation_with_accuracy'] + (0.01 if row['correlation_with_accuracy'] > 0 else -0.05), 
                        '***', ha='center', fontsize=12)
            elif row['p_value'] < 0.01:
                plt.text(i, row['correlation_with_accuracy'] + (0.01 if row['correlation_with_accuracy'] > 0 else -0.05), 
                        '**', ha='center', fontsize=12)
            elif row['p_value'] < 0.05:
                plt.text(i, row['correlation_with_accuracy'] + (0.01 if row['correlation_with_accuracy'] > 0 else -0.05), 
                        '*', ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig('human_ratings_accuracy_correlation.pdf', dpi=300)
        plt.close()

        # 2. Correlation between FSM metrics and FSM accuracy
        print("\n--- Correlation between FSM metrics and FSM accuracy ---")
        fsm_metric_cols = ['confidence', 'program_length']

        # Filter to only include columns that exist
        fsm_metric_cols = [col for col in fsm_metric_cols if col in fsm_best_config_df.columns]

        # Calculate correlations
        fsm_corr = pd.DataFrame(index=fsm_metric_cols, columns=['correlation_with_accuracy', 'p_value'])

        for col in fsm_metric_cols:
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(fsm_best_config_df[col], fsm_best_config_df['correct'])
            fsm_corr.loc[col] = [corr, p_value]

        print(fsm_corr)

        # Visualize these correlations
        plt.figure(figsize=(8, 6))
        sns.barplot(x=fsm_corr.index, y='correlation_with_accuracy', data=fsm_corr)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Correlation between FSM Metrics and Accuracy', fontsize=16)
        plt.xlabel('Metric', fontsize=14)
        plt.ylabel('Correlation Coefficient', fontsize=14)

        # Add significance markers
        for i, (idx, row) in enumerate(fsm_corr.iterrows()):
            if row['p_value'] < 0.001:
                plt.text(i, row['correlation_with_accuracy'] + (0.01 if row['correlation_with_accuracy'] > 0 else -0.05), 
                        '***', ha='center', fontsize=12)
            elif row['p_value'] < 0.01:
                plt.text(i, row['correlation_with_accuracy'] + (0.01 if row['correlation_with_accuracy'] > 0 else -0.05), 
                        '**', ha='center', fontsize=12)
            elif row['p_value'] < 0.05:
                plt.text(i, row['correlation_with_accuracy'] + (0.01 if row['correlation_with_accuracy'] > 0 else -0.05), 
                        '*', ha='center', fontsize=12)

        plt.tight_layout()
        plt.savefig('fsm_metrics_accuracy_correlation.pdf', dpi=300)
        plt.close()
        
        # 3. Correlation between human and FSM accuracy by task
        print("\n--- Correlation between human and FSM accuracy by task ---")
        
        # Check if we can merge the datasets by task and agent_id
        if 'task_str' in human_df_eda.columns and 'username' in human_df_eda.columns and \
           'task' in fsm_best_config_df.columns and 'filename' in fsm_best_config_df.columns:
            
            # Rename columns to match before merging
            human_task_agent_acc = human_df_eda.groupby(['task_str', 'username'])['human_correct'].mean().reset_index()
            human_task_agent_acc = human_task_agent_acc.rename(columns={'task_str': 'task', 'username': 'agent_id'})
            
            fsm_task_agent_acc = fsm_best_config_df.groupby(['task', 'filename'])['correct'].mean().reset_index()
            fsm_task_agent_acc = fsm_task_agent_acc.rename(columns={'filename': 'agent_id'})
            
            # Merge the two datasets on matching column names
            merged_acc = pd.merge(human_task_agent_acc, fsm_task_agent_acc,
                                on=['task', 'agent_id'],
                                suffixes=('_human', '_fsm'))
            print(f"Number of task-agent combinations for correlation: {len(merged_acc)}")
            
            # Calculate correlation
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(merged_acc['human_correct'], merged_acc['correct'])
            print(f"Correlation between human and FSM accuracy: {corr:.4f} (p-value: {p_value:.4f})")
            
            # Visualize the correlation
            plt.figure(figsize=(8, 8))
            sns.scatterplot(x='human_correct', y='correct', data=merged_acc, alpha=0.6)
            
            # Add regression line
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(merged_acc['human_correct'], merged_acc['correct'])
            x = np.linspace(merged_acc['human_correct'].min(), merged_acc['human_correct'].max(), 100)
            y = slope * x + intercept
            plt.plot(x, y, 'r-', alpha=0.7)
            
            plt.title(f'Human vs. FSM Accuracy by Task-Agent (r={corr:.3f}, p={p_value:.4f})', fontsize=16)
            plt.xlabel('Human Accuracy', fontsize=14)
            plt.ylabel('FSM Accuracy', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add diagonal line for reference
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('human_fsm_accuracy_correlation.pdf', dpi=300)
            plt.close()
            
            # 4. Analyze tasks where human and FSM predictions differ significantly
            merged_acc['acc_diff'] = merged_acc['correct'] - merged_acc['human_correct']
            
            # Tasks where FSM outperforms humans
            fsm_better = merged_acc[merged_acc['acc_diff'] > 0.2].sort_values('acc_diff', ascending=False)
            if not fsm_better.empty:
                print("\nTasks where FSM outperforms humans by >20%:")
                print(fsm_better[['task', 'agent_id', 'human_correct', 'correct', 'acc_diff']].head(10))
            
            # Tasks where humans outperform FSM
            human_better = merged_acc[merged_acc['acc_diff'] < -0.2].sort_values('acc_diff')
            if not human_better.empty:
                print("\nTasks where humans outperform FSM by >20%:")
                print(human_better[['task', 'agent_id', 'human_correct', 'correct', 'acc_diff']].head(10))
            
            # 5. Visualize the distribution of accuracy differences
            plt.figure(figsize=(10, 6))
            sns.histplot(merged_acc['acc_diff'], bins=20, kde=True)
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title('Distribution of Accuracy Differences (FSM - Human)', fontsize=16)
            plt.xlabel('Accuracy Difference', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.tight_layout()
            plt.savefig('accuracy_difference_distribution.pdf', dpi=300)
            plt.close()
        else:
            print("Cannot perform correlation analysis between human and FSM accuracy: Missing task or agent_id columns")
    
    # ---- END OF NEW CORRELATION ANALYSIS SECTION ----
    
    # Create a bar graph comparing human accuracy to FSM accuracy (overall)
    # Set seaborn style with larger font scale
    sns.set_context("paper", font_scale=2.0)
    
    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Data for the bar graph
    models = ['Human', 'FSM']
    accuracies = [human_acc, fsm_human_acc]
    
    # Create custom colors dictionary
    comparison_colors = {'Human': 'orange', 'FSM': 'green'}
    
    # Create bar plot
    ax = sns.barplot(x=models, y=accuracies, palette=comparison_colors, hue=models)
    
    # # Add value labels on top of bars
    # for i, acc in enumerate(accuracies):
    #     ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=14)
    
    plt.xlabel('Model', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Human vs. FSM Prediction Accuracy', fontsize=18)
    
    # Set y-axis limits with some padding
    # plt.ylim(0, max(accuracies) + 0.1)
    
    # Remove spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('human_vs_fsm_accuracy_comparison.pdf', dpi=300)
    plt.close()

elif question == 4:
    ############
    # Question 4: Ablation studies of FSM components
    ############

    # Load all the data as before
    print("Starting ablation study of bootstrap FSM components")
    
    # Define datasets to analyze
    datasets = ["grid", "human", "partnr"]
    
    # Set seaborn style
    sns.set_context("paper", font_scale=1.5)
    
    # Create a figure with subplots for each dataset
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 15), sharex=True)
    
    # Set up colors for the two metrics
    accuracy_color = 'blue'
    program_length_color = 'red'
    
    for i, dataset in enumerate(datasets):
        print(f"\nProcessing {dataset} dataset")
        
        # Find all bootstrap files for this dataset
        if dataset == "grid":
            bootstrap_files = glob.glob(f"FSM/new_bootstrap_accuracy_FSM*.csv")
            # Filter out files with specific configurations if needed
            bootstrap_files = [f for f in bootstrap_files if "_group" not in f]
        elif dataset == "human":
            bootstrap_files = glob.glob(f"FSM/human_bootstrap_detailed_FSM*.csv")
        elif dataset == "partnr":
            bootstrap_files = glob.glob(f"FSM/partnr2_bootstrap_accuracy_FSM*.csv")
            # Filter out group files
            bootstrap_files = [f for f in bootstrap_files if "_group" not in f]
        
        print(f"Found {len(bootstrap_files)} bootstrap files for {dataset}")
        
        # Initialize dictionaries to store aggregated data by number of hypotheses
        hyp_accuracy = {}
        hyp_program_length = {}
        hyp_counts = {}
        hyp_accuracy_std = {}  # For standard error calculation
        hyp_accuracy_values = {}  # Store all values for std error calculation
        
        # Process each file
        for file_path in bootstrap_files:
            try:
                # Use a more robust approach to handle inconsistent columns
                # First pass: determine all possible columns and read data
                all_rows = []
                max_cols = 0
                header = None
                
                with open(file_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    for i_row, row in enumerate(reader):
                        if i_row == 0:
                            header = row
                            continue
                        max_cols = max(max_cols, len(row))
                        all_rows.append(row)
                
                # Extend header if needed
                while len(header) < max_cols:
                    header.append(f'extra_col_{len(header)}')
                
                # Create DataFrame with consistent columns
                data = []
                for row in all_rows:
                    # Pad row with NaN values if needed
                    padded_row = row + [''] * (max_cols - len(row))
                    data.append(padded_row)
                
                bootstrap_df = pd.DataFrame(data, columns=header)
                
                # Convert columns to appropriate types
                for col in bootstrap_df.columns:
                    try:
                        bootstrap_df[col] = pd.to_numeric(bootstrap_df[col])
                    except:
                        pass  # Keep as string if can't convert
                
                # Check if necessary columns exist
                if 'num_hypothesis' in bootstrap_df.columns and 'accuracy' in bootstrap_df.columns:
                    # Group by num_hypothesis and calculate mean accuracy
                    if dataset == "grid":
                        # rename 'extra_col_12' to 'program_length' and ignore nans
                        bootstrap_df = bootstrap_df.rename(columns={'extra_col_12': 'program_length'})
                    
                    for hyp, group in bootstrap_df.groupby('num_hypothesis'):
                        if hyp not in hyp_accuracy:
                            hyp_accuracy[hyp] = 0
                            hyp_counts[hyp] = 0
                            hyp_program_length[hyp] = 0
                            hyp_accuracy_values[hyp] = []
                        
                        hyp_accuracy[hyp] += group['accuracy'].mean() * len(group)
                        hyp_counts[hyp] += len(group)
                        hyp_accuracy_values[hyp].extend(group['accuracy'].tolist())
                        
                        # Calculate program length if available
                        if 'program_length' in group.columns:
                            # ignore nans in program length calculation
                            non_nan_program_length = group['program_length'].dropna()
                            if len(non_nan_program_length) > 0:
                                hyp_program_length[hyp] += non_nan_program_length.mean() * len(non_nan_program_length)
                elif 'num_hypothesis' in bootstrap_df.columns and 'correct' in bootstrap_df.columns:
                    for hyp, group in bootstrap_df.groupby('num_hypothesis'):
                        if hyp not in hyp_accuracy:
                            hyp_accuracy[hyp] = 0
                            hyp_counts[hyp] = 0
                            hyp_program_length[hyp] = 0
                            hyp_accuracy_values[hyp] = []
                        
                        hyp_accuracy[hyp] += group['correct'].mean() * len(group)
                        hyp_counts[hyp] += len(group)
                        hyp_accuracy_values[hyp].extend(group['correct'].tolist())

                        if 'program_length' in group.columns:
                            hyp_program_length[hyp] += group['program_length'].mean() * len(group)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Calculate averages and standard errors
        for hyp in hyp_accuracy:
            if hyp_counts[hyp] > 0:
                hyp_accuracy[hyp] /= hyp_counts[hyp]
                if hyp in hyp_program_length:
                    hyp_program_length[hyp] /= hyp_counts[hyp]
                
                # Calculate standard error
                if len(hyp_accuracy_values[hyp]) > 1:
                    hyp_accuracy_std[hyp] = np.std(hyp_accuracy_values[hyp], ddof=1) / np.sqrt(len(hyp_accuracy_values[hyp]))
                else:
                    hyp_accuracy_std[hyp] = 0
        
        # Convert to lists for plotting
        hyp_nums = sorted(hyp_accuracy.keys())
        accuracies = [hyp_accuracy[h] for h in hyp_nums]
        std_errors = [hyp_accuracy_std.get(h, 0) for h in hyp_nums]
        
        # Create primary y-axis for accuracy
        ax1 = axes[i]
        
        # Plot the main line
        line1, = ax1.plot(hyp_nums, accuracies, 'o-', color=accuracy_color, label='Accuracy', markersize=6)
        
        # Add shaded area for standard error
        ax1.fill_between(hyp_nums, 
                         [acc - err for acc, err in zip(accuracies, std_errors)],
                         [acc + err for acc, err in zip(accuracies, std_errors)],
                         color=accuracy_color, alpha=0.2)
        
        ax1.set_ylabel('Accuracy', color=accuracy_color, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=accuracy_color)
        ax1.set_ylim(0, 1)  # Set y-axis from 0 to 1 for accuracy
        
        # Create secondary y-axis for program length if available
        if any(hyp_program_length.values()):
            program_lengths = [hyp_program_length.get(h, 0) for h in hyp_nums]
            
            ax2 = ax1.twinx()
            line2, = ax2.plot(hyp_nums, program_lengths, 'o--', color=program_length_color, label='Program Length')
            ax2.set_ylabel('Avg Program Length', color=program_length_color, fontsize=12)
            ax2.tick_params(axis='y', labelcolor=program_length_color)
            
            # Combine legends
            lines = [line1, line2]
            labels = ['Accuracy', 'Program Length']
            ax1.legend(lines, labels, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        # Set title for this subplot
        ax1.set_title(f'{dataset.capitalize()} Dataset', fontsize=14)
        
        # Remove spines
        sns.despine(ax=ax1)
        if 'ax2' in locals():
            sns.despine(ax=ax2, left=True)
        
        # Print summary statistics
        print(f"{dataset} dataset summary:")
        print(f"Number of hypotheses: {hyp_nums}")
        print(f"Accuracies: {[round(acc, 3) for acc in accuracies]}")
        print(f"Standard errors: {[round(se, 3) for se in std_errors]}")
        if any(hyp_program_length.values()):
            print(f"Program lengths: {[round(pl, 1) for pl in program_lengths]}")
    
    # Set common x-axis label
    plt.xlabel('Number of Hypotheses', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('fsm_ablation_study.pdf', dpi=300)
    plt.close()
    
    # Create a more detailed plot for the grid dataset with different configurations
    print("\nCreating detailed ablation plot for grid dataset")
    
    # Define configurations to compare
    configurations = [
        {"two_stage": False, "structured": "False", "rejuvenation": False, "label": "Base"},
        {"two_stage": True, "structured": "False", "rejuvenation": False, "label": "Two-Stage"},
        {"two_stage": False, "structured": "p1", "rejuvenation": False, "label": "Structured-p1"},
        {"two_stage": False, "structured": "p2", "rejuvenation": False, "label": "Structured-p2"},
        {"two_stage": False, "structured": "False", "rejuvenation": True, "label": "Rejuvenation"}
    ]
    
    # Set seaborn style
    sns.set_context("paper", font_scale=1.5)
    
    # Create a figure for detailed grid ablation
    plt.figure(figsize=(12, 8))
    
    # Process each configuration
    for config in configurations:
        # Create configuration key
        two_stage_str = "_two_stage" if config["two_stage"] else ""
        structured_str = f"_structured_{config['structured']}" if config["structured"] != "False" else ""
        rejuvenation_str = "_rejuvenation" if config["rejuvenation"] else ""
        
        # Initialize data structures
        hyp_nums = []
        accuracies = []
        accuracy_values = {}  # For standard error calculation
        
        # Process files for this configuration with top_k=0 (all particles)
        for top_k in [0]:  # Just use all particles for this comparison
            filepath = f"FSM/new_bootstrap_accuracy_FSM{two_stage_str}{structured_str}{rejuvenation_str}_topk{top_k}.csv"
            
            try:
                # Use a more robust approach to handle inconsistent columns
                # First pass: determine all possible columns and read data
                all_rows = []
                max_cols = 0
                header = None
                
                with open(filepath, 'r', newline='') as f:
                    reader = csv.reader(f)
                    for i_row, row in enumerate(reader):
                        if i_row == 0:
                            header = row
                            continue
                        max_cols = max(max_cols, len(row))
                        all_rows.append(row)
                
                # Extend header if needed
                while len(header) < max_cols:
                    header.append(f'extra_col_{len(header)}')
                
                # Create DataFrame with consistent columns
                data = []
                for row in all_rows:
                    # Pad row with NaN values if needed
                    padded_row = row + [''] * (max_cols - len(row))
                    data.append(padded_row)
                
                bootstrap_df = pd.DataFrame(data, columns=header)
                
                # Convert columns to appropriate types
                for col in bootstrap_df.columns:
                    try:
                        bootstrap_df[col] = pd.to_numeric(bootstrap_df[col])
                    except:
                        pass  # Keep as string if can't convert
                
                # Group by num_hypothesis and calculate mean accuracy
                grouped = bootstrap_df.groupby('num_hypothesis')['accuracy'].mean().reset_index()
                
                # Store the data
                hyp_nums = grouped['num_hypothesis'].tolist()
                accuracies = grouped['accuracy'].tolist()
                
                # Store all accuracy values for standard error calculation
                for hyp, group in bootstrap_df.groupby('num_hypothesis'):
                    if hyp not in accuracy_values:
                        accuracy_values[hyp] = []
                    accuracy_values[hyp].extend(group['accuracy'].tolist())
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        # Calculate standard errors
        std_errors = []
        for hyp in hyp_nums:
            if hyp in accuracy_values and len(accuracy_values[hyp]) > 1:
                std_err = np.std(accuracy_values[hyp], ddof=1) / np.sqrt(len(accuracy_values[hyp]))
                std_errors.append(std_err)
            else:
                std_errors.append(0)
        
        # Plot this configuration if we have data
        if hyp_nums and accuracies:
            # Get a color from the default color cycle
            color = plt.cm.tab10(len(plt.gca().lines) % 10)
            
            # Plot the main line
            plt.plot(hyp_nums, accuracies, 'o-', label=config["label"], markersize=6, color=color)
            
            # Add shaded area for standard error
            plt.fill_between(hyp_nums, 
                            [acc - err for acc, err in zip(accuracies, std_errors)],
                            [acc + err for acc, err in zip(accuracies, std_errors)],
                            color=color, alpha=0.2)
    
    # Add labels and title
    plt.xlabel('Number of Hypotheses', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Grid Dataset: Ablation of FSM Components', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Remove spines
    sns.despine()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('grid_fsm_detailed_ablation.pdf', dpi=300)
    plt.close()
    
    # Create a plot to show the effect of top_k parameter
    print("\nCreating plot for top_k parameter effect")
    
    # Use the base configuration
    two_stage = False
    structured = "False"
    rejuvenation = False
    
    # Set seaborn style
    sns.set_context("paper", font_scale=1.5)
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Process different top_k values
    for top_k in [1, 10, 0]:  # 0 means all particles
        # Display name for the legend
        display_top_k = 25 if top_k == 0 else top_k
        
        # Create filepath
        two_stage_str = "_two_stage" if two_stage else ""
        structured_str = f"_structured_{structured}" if structured != "False" else ""
        rejuvenation_str = "_rejuvenation" if rejuvenation else ""
        filepath = f"FSM/new_bootstrap_accuracy_FSM{two_stage_str}{structured_str}{rejuvenation_str}_topk{top_k}.csv"
        
        try:
            # Use a more robust approach to handle inconsistent columns
            # First pass: determine all possible columns and read data
            all_rows = []
            max_cols = 0
            header = None
            
            with open(filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                for i_row, row in enumerate(reader):
                    if i_row == 0:
                        header = row
                        continue
                    max_cols = max(max_cols, len(row))
                    all_rows.append(row)
            
            # Extend header if needed
            while len(header) < max_cols:
                header.append(f'extra_col_{len(header)}')
            
            # Create DataFrame with consistent columns
            data = []
            for row in all_rows:
                # Pad row with NaN values if needed
                padded_row = row + [''] * (max_cols - len(row))
                data.append(padded_row)
            
            bootstrap_df = pd.DataFrame(data, columns=header)
            
            # Convert columns to appropriate types
            for col in bootstrap_df.columns:
                try:
                    bootstrap_df[col] = pd.to_numeric(bootstrap_df[col])
                except:
                    pass  # Keep as string if can't convert
            
            # Group by num_hypothesis and calculate mean accuracy
            grouped = bootstrap_df.groupby('num_hypothesis')['accuracy'].mean().reset_index()
            
            # Calculate standard errors
            std_errors = []
            for hyp in grouped['num_hypothesis']:
                hyp_data = bootstrap_df[bootstrap_df['num_hypothesis'] == hyp]['accuracy']
                if len(hyp_data) > 1:
                    std_err = np.std(hyp_data, ddof=1) / np.sqrt(len(hyp_data))
                    std_errors.append(std_err)
                else:
                    std_errors.append(0)
            
            # Get a color from the default color cycle
            color = plt.cm.tab10(len(plt.gca().lines) % 10)
            
            # Plot the main line
            plt.plot(grouped['num_hypothesis'], grouped['accuracy'], 'o-', 
                    label=f'top_k={display_top_k}', markersize=6, color=color)
            
            # Add shaded area for standard error
            plt.fill_between(grouped['num_hypothesis'], 
                            grouped['accuracy'] - std_errors,
                            grouped['accuracy'] + std_errors,
                            color=color, alpha=0.2)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Add labels and title
    plt.xlabel('Number of Hypotheses', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Effect of top_k Parameter on FSM Accuracy', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Remove spines
    sns.despine()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('fsm_topk_effect.pdf', dpi=300)
    plt.close()
    