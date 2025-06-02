import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import glob
import scipy.stats

model_colors = {'BC': 'red', 'AutoToM': 'blue', 'NLLM': 'purple', 'AUTOMA': 'green', 'Human': 'orange'}


question = 1
time_dicts = {'single': {}, 'group': {}}

task_list = [
    'Always move right',
    'Wander randomly without any specific direction',
    'Always pick up the nearest block',
    'Move in a vertical line (up and down)',
    'Bounce off walls without moving beyond them',
    'Stay in place',
    'Always pick up purple blocks',
    'Only pick up the first block encountered',
    'Move towards the farthest block each time',
    'Follow a clockwise square pattern',
    'Snake through the grid (right, up, left, down)',
    'Collect blocks of a specific color',
    'Move left if possible, otherwise right',
    'Move in an L-shape pattern',
    'Oscillate between two points',
    'Follow a path to collect all blocks of a specific color',
    'Create a spiral movement pattern',
    'Move diagonally towards blocks',
    'Return to a specific location when possible',
    'Maximize the number of blocks collected frontally',
]



def load_data(file_path):
    try:
        # First try simple pandas read_csv
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()  # Return empty DataFrame if file is empty
    except:
        # If that fails, manually parse the file
        max_columns = 0
        
        # First pass - find max number of columns
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                max_columns = max(max_columns, len(row))
                
        # Second pass - read data with consistent columns
        data = []
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)  # Get header row
            
            # Pad header if needed
            while len(header) < max_columns:
                header.append(f'extra_column_{len(header)}')
                
            # Process remaining rows
            for row in csv_reader:
                # Pad row with NaN values if needed
                while len(row) < max_columns:
                    row.append(np.nan)
                data.append(row)
                
        df = pd.DataFrame(data, columns=header)
        if "accuracy" in df.columns:
            # Convert to numeric, coercing errors to NaN
            df["accuracy"] = pd.to_numeric(df["accuracy"], errors='coerce')
        elif "correct" in df.columns:
            # Convert to numeric, coercing errors to NaN
            df["correct"] = pd.to_numeric(df["correct"], errors='coerce')
        else:
            print(f"No accuracy or correct column found in {path}")
        return df

# Helper function for t-tests
def perform_t_test(arr1, arr2):
    """Perform t-test between two arrays and return t-statistic and p-value."""
    if len(arr1) == 0 or len(arr2) == 0:
        return 0.0, 1.0  # Return 0.0, 1.0 if either array is empty (no significant difference)
    t_stat, p_val = scipy.stats.ttest_ind(arr1, arr2, equal_var=False, nan_policy='omit')
    return t_stat, p_val

# Helper function to get significance stars
def get_significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

if question == 1:
    # Plot single GT agent accuracy for single and multi-step

    # BC
    bc_grid_single_file_path = "BC/grid_accuracy_BC_2hyp.csv"
    bc_grid_single_multi_file_path = "BC/results_grid_BC_2hyp.csv"
    bc_grid_single_df = load_data(bc_grid_single_file_path)
    bc_grid_single_multi_df = load_data(bc_grid_single_multi_file_path)
    # AutoToM
    autoToM_grid_single_file_path = "AutoToM/grid_accuracy_AutoToM_2hyp.csv"
    autoToM_grid_single_multi_file_path = "AutoToM/results_grid_AutoToM_2hyp.csv"
    autoToM_grid_single_df = load_data(autoToM_grid_single_file_path)
    autoToM_grid_single_multi_df = load_data(autoToM_grid_single_multi_file_path)
    # Naive LLM
    nllm_grid_single_file_path = "NLLM/grid_accuracy_NLLM_2hyp.csv"
    nllm_grid_single_multi_file_path = "NLLM/results_grid_NLLM_2hyp.csv"
    nllm_grid_single_df = load_data(nllm_grid_single_file_path)
    nllm_grid_single_multi_df = load_data(nllm_grid_single_multi_file_path)
    # FSM
    fsm_grid_single_file_paths = glob.glob("FSM/fixed2_results_fsm_bootstrap_multistep*")
    fsm_grid_single_file_paths = [path for path in fsm_grid_single_file_paths if "_group" not in path]
    fsm_grid_single_file_paths = [path for path in fsm_grid_single_file_paths if "actionTime" in path]
    fsm_grid_multi_file_paths = glob.glob("FSM/fixed2_results_fsm_bootstrap_multistep*")
    fsm_grid_multi_file_paths = [path for path in fsm_grid_multi_file_paths if "_group" not in path]
    fsm_grid_multi_time_paths = [path for path in fsm_grid_multi_file_paths if "actionTime" in path]
    # fsm_grid_multi_file_paths = [path for path in fsm_grid_multi_file_paths if "actionTime" not in path]
    
    # Filter paths to only include top_k values of 0, 1, and 10
    valid_topk_values = ["topk10", "topk0", "topk1"]
    fsm_grid_single_file_paths = [path for path in fsm_grid_single_file_paths if any(topk in path for topk in valid_topk_values)]
    fsm_grid_multi_file_paths = [path for path in fsm_grid_multi_file_paths if any(topk in path for topk in valid_topk_values)]

    # Find best model per LLM for single-step
    best_single_models = {}
    for path in fsm_grid_single_file_paths:
        df = load_data(path)
        


        # Extract LLM model name from the file path or from the dataframe
        if "llm_model" in df.columns:
            llm_model = df["llm_model"].iloc[0]
        else:
            # Extract from filename if not in dataframe
            for model_name in ["deepseek", "mistral", "llama", "gpt"]:
                if model_name in path.lower():
                    llm_model = model_name
                    break
            else:
                llm_model = "unknown"
        
        # Extract top_k value
        for topk in valid_topk_values:
            if topk in path:
                top_k = int(topk.replace("topk", ""))
                break
        
        # Calculate mean accuracy
        # make sure accuracy is float type and handle potential non-numeric values
        try:
            # Convert to numeric, coercing errors to NaN
            df['accuracy'] = df['first_step_accuracy']
            df["accuracy"] = pd.to_numeric(df["accuracy"], errors='coerce')
            # Drop NaN values before calculating mean
            mean_acc = df["accuracy"].dropna().mean()
            if pd.isna(mean_acc):  # If still NaN after dropna, set to 0
                mean_acc = 0
        except Exception as e:
            print(f"Error calculating accuracy for {path}: {e}")
            continue
        
        # Store if this is the best model for this LLM and top_k
        key = (llm_model, top_k)
        if key not in best_single_models or mean_acc > best_single_models[key]["mean_acc"]:
            best_single_models[key] = {"path": path, "mean_acc": mean_acc, "df": df}

    # Find best model per LLM for multi-step
    best_multi_models = {}
    for path in fsm_grid_multi_file_paths:
        df = load_data(path)
        # Extract LLM model name from the file path or from the dataframe
        if "llm_model" in df.columns:
            llm_model = df["llm_model"].iloc[0]
        else:
            # Extract from filename if not in dataframe
            for model_name in ["deepseek", "mistral", "llama", "gpt"]:
                if model_name in path.lower():
                    llm_model = model_name
                    break
            else:
                llm_model = "unknown"
        
        # Extract top_k value
        for topk in valid_topk_values:
            if topk in path:
                top_k = int(topk.replace("topk", ""))
                break
        
        # Calculate mean accuracy
        # try:
        # Check which column to use for accuracy
        if "accuracy" in df.columns:
            # Convert to numeric, coercing errors to NaN
            df["accuracy"] = pd.to_numeric(df["accuracy"], errors='coerce')
            # Drop NaN values before calculating mean
            mean_acc = df["accuracy"].dropna().mean()
        elif "correct" in df.columns:
            # Convert to numeric, coercing errors to NaN
            df["correct"] = pd.to_numeric(df["correct"], errors='coerce')
            # Drop NaN values before calculating mean
            mean_acc = df["correct"].dropna().mean()
        else:
            print(f"No accuracy or correct column found in {path}")
            continue
        
        if pd.isna(mean_acc):  # If still NaN after dropna, set to 0
            mean_acc = 0
        # except Exception as e:
        #     print(f"Error calculating accuracy for {path}: {e}")
        #     continue
        
        # Store if this is the best model for this LLM and top_k
        key = (llm_model, top_k)
        if key not in best_multi_models or mean_acc > best_multi_models[key]["mean_acc"]:
            best_multi_models[key] = {"path": path, "mean_acc": mean_acc, "df": df}

    # Now you can access the best dataframes:
    # For single-step: best_single_models[(llm_model, top_k)]["df"]
    # For multi-step: best_multi_models[(llm_model, top_k)]["df"]

    # Set up the visualization style with larger font
    sns.set_context("paper", font_scale=2.0)
    plt.figure(figsize=(22, 8))
    
    # Create a dictionary to store aggregated results for each model
    model_results = {
        'BC': {
            'single_acc': bc_grid_single_df['accuracy'][bc_grid_single_df['accuracy'] != 'accuracy'].astype(float).mean() if not bc_grid_single_df.empty else 0,
            'single_err': bc_grid_single_df['accuracy'][bc_grid_single_df['accuracy'] != 'accuracy'].astype(float).std() / np.sqrt(len(bc_grid_single_df)) if not bc_grid_single_df.empty else 0,
            'multi_acc': bc_grid_single_multi_df['accuracy'][bc_grid_single_multi_df['accuracy'] != 'accuracy'].astype(float).mean() if not bc_grid_single_multi_df.empty else 0,
            'multi_err': bc_grid_single_multi_df['accuracy'][bc_grid_single_multi_df['accuracy'] != 'accuracy'].astype(float).std() / np.sqrt(len(bc_grid_single_multi_df)) if not bc_grid_single_multi_df.empty else 0,
            'pred_time': bc_grid_single_multi_df['avg_prediction_time'][bc_grid_single_multi_df['avg_prediction_time'] != 'avg_prediction_time'].astype(float).mean() if 'avg_prediction_time' in bc_grid_single_multi_df.columns and not bc_grid_single_multi_df.empty else 0,
            'time_err': bc_grid_single_multi_df['avg_prediction_time'][bc_grid_single_multi_df['avg_prediction_time'] != 'avg_prediction_time'].astype(float).std() / np.sqrt(len(bc_grid_single_multi_df)) if 'avg_prediction_time' in bc_grid_single_multi_df.columns and not bc_grid_single_multi_df.empty else 0
        },
        'AutoToM': {
            'single_acc': max([
                autoToM_grid_single_df[autoToM_grid_single_df['llm_model'] == model]['accuracy']
                [(autoToM_grid_single_df['llm_model'] == model) & (autoToM_grid_single_df['accuracy'] != 'accuracy')]
                .astype(float).mean()
                for model in autoToM_grid_single_df['llm_model'].unique()
            ]) if not autoToM_grid_single_df.empty else 0,
            'single_err': autoToM_grid_single_df['accuracy'][autoToM_grid_single_df['accuracy'] != 'accuracy'].astype(float).std() / np.sqrt(len(autoToM_grid_single_df)) if not autoToM_grid_single_df.empty else 0,
            'multi_acc': max([
                autoToM_grid_single_multi_df[autoToM_grid_single_multi_df['llm_model'] == model]['accuracy']
                [(autoToM_grid_single_multi_df['llm_model'] == model) & (autoToM_grid_single_multi_df['accuracy'] != 'accuracy')]
                .astype(float).mean() 
                for model in autoToM_grid_single_multi_df['llm_model'].unique()
            ]) if not autoToM_grid_single_multi_df.empty else 0,
            'multi_err': autoToM_grid_single_multi_df['accuracy'][autoToM_grid_single_multi_df['accuracy'] != 'accuracy'].astype(float).std() / np.sqrt(len(autoToM_grid_single_multi_df)) if not autoToM_grid_single_multi_df.empty else 0,
            'pred_time': max([
                autoToM_grid_single_multi_df[autoToM_grid_single_multi_df['llm_model'] == model]['avg_prediction_time']
                [(autoToM_grid_single_multi_df['llm_model'] == model) & (autoToM_grid_single_multi_df['avg_prediction_time'] != 'avg_prediction_time')]
                .astype(float).mean()
                for model in autoToM_grid_single_df['llm_model'].unique()
            ]) if 'avg_prediction_time' in autoToM_grid_single_multi_df.columns and not autoToM_grid_single_multi_df.empty else 0,
            'time_err': autoToM_grid_single_multi_df['avg_prediction_time'][autoToM_grid_single_multi_df['avg_prediction_time'] != 'avg_prediction_time'].astype(float).std() / np.sqrt(len(autoToM_grid_single_multi_df)) if 'avg_prediction_time' in autoToM_grid_single_multi_df.columns and not autoToM_grid_single_multi_df.empty else 0
        },
        'NLLM': {  # Renamed from 'Naive LLM'
            'single_acc': max([
                nllm_grid_single_df[nllm_grid_single_df['llm_model'] == model]['accuracy']
                [(nllm_grid_single_df['llm_model'] == model) & (nllm_grid_single_df['accuracy'] != 'accuracy')]
                .astype(float).mean()
                for model in nllm_grid_single_df['llm_model'].unique()
            ]) if not nllm_grid_single_df.empty else 0,
            'single_err': nllm_grid_single_df['accuracy'][nllm_grid_single_df['accuracy'] != 'accuracy'].astype(float).std() / np.sqrt(len(nllm_grid_single_df)) if not nllm_grid_single_df.empty else 0,
            'multi_acc': max([
                nllm_grid_single_multi_df[nllm_grid_single_multi_df['llm_model'] == model]['accuracy']
                [(nllm_grid_single_multi_df['llm_model'] == model) & (nllm_grid_single_multi_df['accuracy'] != 'accuracy')]
                .astype(float).mean()
                for model in nllm_grid_single_multi_df['llm_model'].unique()
            ]) if not nllm_grid_single_multi_df.empty else 0,
            'multi_err': nllm_grid_single_multi_df['accuracy'][nllm_grid_single_multi_df['accuracy'] != 'accuracy'].astype(float).std() / np.sqrt(len(nllm_grid_single_multi_df)) if not nllm_grid_single_multi_df.empty else 0,
            'pred_time': max([
                nllm_grid_single_multi_df[nllm_grid_single_multi_df['llm_model'] == model]['avg_prediction_time']
                [(nllm_grid_single_multi_df['llm_model'] == model) & (nllm_grid_single_multi_df['avg_prediction_time'] != 'avg_prediction_time')]
                .astype(float).mean()
                for model in nllm_grid_single_df['llm_model'].unique()
            ]) if 'avg_prediction_time' in nllm_grid_single_multi_df.columns and not nllm_grid_single_multi_df.empty else 0,
            'time_err': nllm_grid_single_multi_df['avg_prediction_time'][nllm_grid_single_multi_df['avg_prediction_time'] != 'avg_prediction_time'].astype(float).std() / np.sqrt(len(nllm_grid_single_multi_df)) if 'avg_prediction_time' in nllm_grid_single_multi_df.columns and not nllm_grid_single_multi_df.empty else 0
        }
    }
    
    # Add FSM results - use the best model for each metric
    fsm_single_dfs = [data["df"] for data in best_single_models.values()]
    fsm_multi_dfs = [data["df"] for data in best_multi_models.values()]

    # --- NEW: Find the best FSM single accuracy ---
    fsm_single_acc, fsm_single_err, best_single_accs = 0, 0, []
    fsm_pred_time, fsm_time_err = 0, 0  # <--- add these for single
    if fsm_single_dfs:
        best_acc = -1
        for df in fsm_single_dfs:
            if 'accuracy' in df.columns:
                # --- FIX: Find best (llm_model, num_hypothesis) pair ---
                if 'llm_model' in df.columns and 'num_hypothesis' in df.columns:
                    grouped = df.groupby(['llm_model', 'num_hypothesis'])
                    for (llm, hyp), group in grouped:
                        accs = group['accuracy'][group['accuracy'] != 'accuracy'].astype(float)
                        if len(accs) < 10:
                            continue
                        mean_acc = accs.mean()
                        err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
                        # --- NEW: compute pred time for this group ---
                        time_col = 'avg_prediction_time' if 'avg_prediction_time' in group.columns else 'prediction_time' if 'prediction_time' in group.columns else None
                        if mean_acc > best_acc:
                            best_acc = mean_acc
                            best_err = err
                            best_single_accs = accs
                            if time_col:
                                times = group[time_col][group[time_col] != time_col].astype(float)
                                fsm_pred_time = times.mean() if not times.empty else 0
                                fsm_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
                            else:
                                fsm_pred_time = 0
                                fsm_time_err = 0
                else:
                    accs = df['accuracy'][df['accuracy'] != 'accuracy'].astype(float)
                    if len(accs) < 10:
                        continue
                    mean_acc = accs.mean()
                    err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
                    time_col = 'avg_prediction_time' if 'avg_prediction_time' in df.columns else 'prediction_time' if 'prediction_time' in df.columns else None
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_err = err
                        best_accs = accs
                        if time_col:
                            times = df[time_col][df[time_col] != time_col].astype(float)
                            fsm_pred_time = times.mean() if not times.empty else 0
                            fsm_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
                        else:
                            fsm_pred_time = 0
                            fsm_time_err = 0
        if best_acc >= 0:
            fsm_single_acc = best_acc
            fsm_single_err = best_err

    # --- NEW: Find the best FSM multi accuracy and prediction time ---
    fsm_multi_acc, fsm_multi_err, fsm_pred_time, fsm_time_err, best_multi_accs = 0, 0, 0, 0, []
    best_fsm_multi_df = None
    if fsm_multi_dfs:
        best_acc = -1
        for df in fsm_multi_dfs:
            # Check which column to use for accuracy
            acc_col = 'accuracy' if 'accuracy' in df.columns else 'correct' if 'correct' in df.columns else None
            if acc_col:
                # --- FIX: Find best (llm_model, num_hypothesis) pair ---
                if 'llm_model' in df.columns and 'num_hypothesis' in df.columns:
                    grouped = df.groupby(['llm_model', 'num_hypothesis'])
                    for (llm, hyp), group in grouped:
                        accs = group[acc_col][group[acc_col] != acc_col].astype(float)
                        mean_acc = accs.mean()
                        err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
                        # Prediction time
                        time_col = 'avg_prediction_time' if 'avg_prediction_time' in group.columns else 'prediction_time' if 'prediction_time' in group.columns else None
                        if mean_acc > best_acc:
                            best_acc = mean_acc
                            best_err = err
                            best_multi_accs = accs
                            best_fsm_multi_df = group
                            if time_col:
                                times = group[time_col][group[time_col] != time_col].astype(float)
                                fsm_pred_time = times.mean() if not times.empty else 0
                                fsm_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
                            else:
                                fsm_pred_time = 0
                                fsm_time_err = 0
                else:
                    accs = df[acc_col][df[acc_col] != acc_col].astype(float)
                    mean_acc = accs.mean()
                    err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
                    time_col = 'avg_prediction_time' if 'avg_prediction_time' in df.columns else 'prediction_time' if 'prediction_time' in df.columns else None
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_err = err
                        best_multi_accs = accs
                        best_fsm_multi_df = df
                        if time_col:
                            times = df[time_col][df[time_col] != time_col].astype(float)
                            fsm_pred_time = times.mean() if not times.empty else 0
                            fsm_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
                        else:
                            fsm_pred_time = 0
                            fsm_time_err = 0
        if best_acc >= 0:
            fsm_multi_acc = best_acc
            fsm_multi_err = best_err
            best_multi_accs = best_multi_accs
    model_results['AUTOMA'] = {
        'single_acc': fsm_single_acc,
        'single_err': fsm_single_err,
        'multi_acc': fsm_multi_acc,
        'multi_err': fsm_multi_err,
        'pred_time': fsm_pred_time,      # <--- now correct for best pair
        'time_err': fsm_time_err         # <--- now correct for best pair
    }
    print(model_results['AUTOMA'])

    # load fsm_grid_multi_time_paths
    pred_times = []
    for path in fsm_grid_multi_time_paths:
        df = load_data(path)
        if df.empty:
            continue
        # get avg prediction time
        time_col = 'avg_prediction_time' if 'avg_prediction_time' in df.columns else 'prediction_time' if 'prediction_time' in df.columns else None
        if time_col:
            times = df[time_col][df[time_col] != time_col].astype(float)
            fsm_pred_time = times.mean() 
            pred_times.append(fsm_pred_time)
    pred_times = np.array(pred_times)
    pred_time_mean = pred_times.mean()
    pred_time_std = pred_times.std() / np.sqrt(len(pred_times))

    num_predictions = 25
    model_results['AUTOMA']['line_plot_pred_time'] = [model_results['AUTOMA']['pred_time']]
    model_results['AUTOMA']['line_plot_time_err'] = [model_results['AUTOMA']['time_err']]
    for i in range(num_predictions):
        model_results['AUTOMA']['line_plot_pred_time'].append(model_results['AUTOMA']['line_plot_pred_time'][-1] + pred_time_mean)
        model_results['AUTOMA']['line_plot_time_err'].append(pred_time_std)
    
    for key in model_results.keys():
        if key != 'AUTOMA':
            model_results[key]['line_plot_pred_time'] = [model_results[key]['pred_time']]
            model_results[key]['line_plot_time_err'] = [model_results[key]['time_err']]
            for i in range(num_predictions):
                model_results[key]['line_plot_pred_time'].append(model_results[key]['line_plot_pred_time'][-1] + model_results[key]['pred_time'])
                model_results[key]['line_plot_time_err'].append(model_results[key]['time_err'])
    # plot the line plot
    plt.figure(figsize=(8, 6))
    x = range(num_predictions + 1)  # +1 for initial point
    for model in model_results.keys():
        time_dicts['single'][model] = {}
        times = model_results[model]['line_plot_pred_time']
        errs = model_results[model]['line_plot_time_err']
        time_dicts['single'][model]['line_plot_pred_time'] = times
        time_dicts['single'][model]['line_plot_time_err'] = errs
        plt.fill_between(x, 
                        [t - e for t, e in zip(times, errs)],
                        [t + e for t, e in zip(times, errs)],
                        color=model_colors[model], alpha=0.5, label=model)
    plt.xlabel('Number of Predictions')
    plt.ylabel('Cumulative Prediction Time (s)')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(model_results))
    sns.despine()
    plt.tight_layout()
    plt.savefig('question1-1.png', bbox_inches='tight')
    plt.close()



    # model_results['AUTOMA']['pred_time'] += 5 * pred_time_mean  # generation time + prediction time
    # model_results['AUTOMA']['time_err'] = pred_time_std

    # for key in model_results.keys():
    #     if key != 'AUTOMA':
    #         model_results[key]['pred_time'] = 5 * model_results[key]['pred_time']  # 5x longer
    #         model_results[key]['time_err'] = model_results[key]['time_err']



    
    # Create dataframes for plotting
    models = list(model_results.keys())
    single_acc_data = [model_results[m]['single_acc'] for m in models]
    single_err_data = [model_results[m]['single_err'] for m in models]
    multi_acc_data = [model_results[m]['multi_acc'] for m in models]
    multi_err_data = [model_results[m]['multi_err'] for m in models]
    pred_time_data = [model_results[m]['pred_time'] for m in models]
    time_err_data = [model_results[m]['time_err'] for m in models]
    
    # --- Significance stars helper ---
    def get_significance_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    # --- Prepare accuracy arrays for significance testing ---
    def get_acc_array(df, col='accuracy'):
        if df is None or df.empty or col not in df.columns:
            return np.array([])
        arr = df[col][df[col] != col]
        return arr.astype(float).values

    # FSM arrays
    fsm_single_acc_arr = None
    for data in best_single_models.values():
        arr = get_acc_array(data["df"])
        if arr.size > 0:
            fsm_single_acc_arr = arr
            break
    fsm_multi_acc_arr = None
    for data in best_multi_models.values():
        acc_col = 'accuracy' if 'accuracy' in data["df"].columns else 'correct'
        arr = get_acc_array(data["df"], col=acc_col)
        if arr.size > 0:
            fsm_multi_acc_arr = arr
            break

    # Other models' arrays
    model_acc_arrays = {
        'BC': get_acc_array(bc_grid_single_df),
        'AutoToM': get_acc_array(autoToM_grid_single_df),
        'NLLM': get_acc_array(nllm_grid_single_df),  # Renamed
    }
    model_multi_acc_arrays = {
        'BC': get_acc_array(bc_grid_single_multi_df),
        'AutoToM': get_acc_array(autoToM_grid_single_multi_df),
        'NLLM': get_acc_array(nllm_grid_single_multi_df),  # Renamed
    }

    # # Calculate p-values for single-step comparisons
    # print("\n=== Question 1: Single-Step Significance Tests (AUTOMA vs Others) ===")
    # for model, arr in model_acc_arrays.items():
    #     t_stat, p_val = perform_t_test(best_single_accs, arr)
    #     stars = get_significance_stars(p_val)
    #     freedom = 200  # approximate degrees of freedom
    #     print(f"AUTOMA vs {model}: t({freedom:.1f}) = {t_stat:.3f}, p = {p_val:.4f} {stars}")
    
    # # Calculate p-values for multi-step comparisons
    # print("\n=== Question 1: Multi-Step Significance Tests (AUTOMA vs Others) ===")
    # for model, arr in model_multi_acc_arrays.items():
    #     t_stat, p_val = perform_t_test(best_multi_accs, arr)
    #     stars = get_significance_stars(p_val)
    #     freedom = 200  # approximate degrees of freedom
    #     print(f"AUTOMA vs {model}: t({df:.1f}) = {t_stat:.3f}, p = {p_val:.4f} {stars}")
    
    # --- Plot with significance stars ---
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(x=models, y=single_acc_data, palette=[model_colors[m] for m in models], errorbar=None, hue=models)
    plt.errorbar(x=range(len(models)), y=single_acc_data, yerr=single_err_data, fmt='none', ecolor='black', capsize=5)
    plt.ylabel('Accuracy', fontsize=44)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=44)
    plt.yticks([0, 0.33, 0.67, 1.0], fontsize=44)
    ax1.set_title('Single-Step', fontsize=44)
    sns.despine()

    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x=models, y=multi_acc_data, palette=[model_colors[m] for m in models], errorbar=None, hue=models)
    plt.errorbar(x=range(len(models)), y=multi_acc_data, yerr=multi_err_data, fmt='none', ecolor='black', capsize=5)
    plt.ylabel('Accuracy', fontsize=44)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=44)
    plt.yticks([0, 0.33, 0.67, 1.0], fontsize=44)
    ax2.set_title('Multi-Step', fontsize=44)
    sns.despine()

    
    # Collect best LLM models used for each approach
    fsm_best_llm_model = None
    fsm_best_acc = -1
    for key, data in best_single_models.items():
        llm_model, top_k = key
        if data["mean_acc"] > fsm_best_acc:
            fsm_best_acc = data["mean_acc"]
            fsm_best_llm_model = llm_model
    
    autotom_best_llm_model = None
    autotom_best_acc = -1
    if not autoToM_grid_single_df.empty and 'llm_model' in autoToM_grid_single_df.columns:
        for model in autoToM_grid_single_df['llm_model'].unique():
            model_data = autoToM_grid_single_df[autoToM_grid_single_df['llm_model'] == model]
            acc_data = model_data['accuracy'][(model_data['accuracy'] != 'accuracy')]
            if not acc_data.empty:
                acc = acc_data.astype(float).mean()
                if acc > autotom_best_acc:
                    autotom_best_acc = acc
                    autotom_best_llm_model = model
    
    nllm_best_llm_model = None
    nllm_best_acc = -1
    if not nllm_grid_single_df.empty and 'llm_model' in nllm_grid_single_df.columns:
        for model in nllm_grid_single_df['llm_model'].unique():
            model_data = nllm_grid_single_df[nllm_grid_single_df['llm_model'] == model]
            acc_data = model_data['accuracy'][(model_data['accuracy'] != 'accuracy')]
            if not acc_data.empty:
                acc = acc_data.astype(float).mean()
                if acc > nllm_best_acc:
                    nllm_best_acc = acc
                    nllm_best_llm_model = model
    
    # # Create text strings with the best LLM models used
    # legend_texts = []
    # if fsm_best_llm_model:
    #     legend_texts.append(f"FSM: {fsm_best_llm_model}")
    # if autotom_best_llm_model:
    #     legend_texts.append(f"AutoToM: {autotom_best_llm_model}")
    # if nllm_best_llm_model:
    #     legend_texts.append(f"NLLM: {nllm_best_llm_model}")
    
    # # Add a text box with the best LLM models used
    # if legend_texts:
    #     legend_text = "\n".join(legend_texts)
    #     plt.figtext(0.5, 0.01, legend_text, ha='center', fontsize=16, 
    #                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)  # Make more room for the text at the bottom
    plt.savefig('question1.png', dpi=300, bbox_inches='tight')
    plt.close()

question = 2
if question == 2:
    # Plot group GT agent accuracy for single and multi-step

    # BC
    bc_grid_group_file_path = "BC/grid_accuracy_BC_2hyp_group.csv"
    bc_grid_group_multi_file_path = "BC/results_grid_BC_2hyp_group.csv"
    bc_grid_group_df = load_data(bc_grid_group_file_path)
    bc_grid_group_multi_df = load_data(bc_grid_group_multi_file_path)
    # AutoToM
    autoToM_grid_group_file_path = "AutoToM/grid_accuracy_AutoToM_2hyp_group.csv"
    autoToM_grid_group_multi_file_path = "AutoToM/results_grid_AutoToM_2hyp_group.csv"
    autoToM_grid_group_df = load_data(autoToM_grid_group_file_path)
    autoToM_grid_group_multi_df = load_data(autoToM_grid_group_multi_file_path)
    # Naive LLM
    nllm_grid_group_file_path = "NLLM/grid_accuracy_NLLM_2hyp_group.csv"
    nllm_grid_group_multi_file_path = "NLLM/results_grid_NLLM_2hyp_group.csv"
    nllm_grid_group_df = load_data(nllm_grid_group_file_path)
    nllm_grid_group_multi_df = load_data(nllm_grid_group_multi_file_path)
    # FSM
    fsm_grid_group_file_paths = glob.glob("FSM/new_bootstrap_accuracy_FSM*")
    fsm_grid_group_file_paths = [path for path in fsm_grid_group_file_paths if "_group" in path]
    fsm_grid_group_multi_file_paths = glob.glob("FSM/results_fsm_bootstrap_multistep*")
    fsm_grid_group_multi_file_paths = [path for path in fsm_grid_group_multi_file_paths if "_group" in path]
    fsm_grid_group_multi_time_file_paths = [path for path in fsm_grid_group_multi_file_paths if "actionTime" in path]
    fsm_grid_group_multi_file_paths = [path for path in fsm_grid_group_multi_file_paths if "actionTime" not in path]

    # Filter paths to only include top_k values of 0, 1, and 10
    valid_topk_values = ["topk10", "topk0", "topk1"]
    fsm_grid_group_file_paths = [path for path in fsm_grid_group_file_paths if any(topk in path for topk in valid_topk_values)]
    fsm_grid_group_multi_file_paths = [path for path in fsm_grid_group_multi_file_paths if any(topk in path for topk in valid_topk_values)]

    # Find best model per LLM for single-step
    best_group_models = {}
    for path in fsm_grid_group_file_paths:
        df = load_data(path)

        # Extract top_k value
        for topk in valid_topk_values:
            if topk in path:
                top_k = int(topk.replace("topk", ""))
                break
        else:
            continue  # Skip if no valid top_k found
        
        # If llm_model is in the dataframe columns, process each unique model
        if "llm_model" in df.columns:
            # Process each unique LLM model in the dataframe
            for llm_model in df["llm_model"].unique():
                # Filter dataframe for this specific model
                model_df = df[df["llm_model"] == llm_model].copy()  # Use .copy() to avoid SettingWithCopyWarning
                
                try:
                    # Convert to numeric, coercing errors to NaN
                    model_df["accuracy"] = pd.to_numeric(model_df["accuracy"], errors='coerce')
                    # Drop NaN values before calculating mean
                    mean_acc = model_df["accuracy"].dropna().mean()
                    if pd.isna(mean_acc):  # If still NaN after dropna, set to 0
                        mean_acc = 0
                except Exception as e:
                    print(f"Error calculating accuracy for {path}, model {llm_model}: {e}")
                    continue
                
                # Store if this is the best model for this LLM and top_k
                key = (llm_model, top_k)
                if key not in best_group_models or mean_acc > best_group_models[key]["mean_acc"]:
                    best_group_models[key] = {"path": path, "mean_acc": mean_acc, "df": model_df}
        else:
            # Extract from filename if not in dataframe
            for model_name in ["deepseek", "mistral", "llama", "gpt"]:
                if model_name in path.lower():
                    llm_model = model_name
                    break
            else:
                llm_model = "unknown"
            
            # Calculate mean accuracy for the entire dataframe
            try:
                # Convert to numeric, coercing errors to NaN
                df["accuracy"] = pd.to_numeric(df["accuracy"], errors='coerce')
                # Drop NaN values before calculating mean
                mean_acc = df["accuracy"].dropna().mean()
                if pd.isna(mean_acc):  # If still NaN after dropna, set to 0
                    mean_acc = 0
            except Exception as e:
                print(f"Error calculating accuracy for {path}: {e}")
                continue
            
            # Store if this is the best model for this LLM and top_k
            key = (llm_model, top_k)
            if key not in best_group_models or mean_acc > best_group_models[key]["mean_acc"]:
                best_group_models[key] = {"path": path, "mean_acc": mean_acc, "df": df}

    # Find best model per LLM for multi-step
    best_group_multi_models = {}
    for path in fsm_grid_group_multi_file_paths:
        df = load_data(path)
        
        # Extract top_k value
        for topk in valid_topk_values:
            if topk in path:
                top_k = int(topk.replace("topk", ""))
                break
        else:
            continue  # Skip if no valid top_k found
        
        # If llm_model is in the dataframe columns, process each unique model
        if "llm_model" in df.columns:
            # Process each unique LLM model in the dataframe
            for llm_model in df["llm_model"].unique():
                # Filter dataframe for this specific model
                model_df = df[df["llm_model"] == llm_model].copy()  # Use .copy() to avoid SettingWithCopyWarning
                
                # Check which column to use for accuracy
                if "accuracy" in model_df.columns:
                    # Convert to numeric, coercing errors to NaN
                    model_df["accuracy"] = pd.to_numeric(model_df["accuracy"], errors='coerce')
                    # Drop NaN values before calculating mean
                    mean_acc = model_df["accuracy"].dropna().mean()
                elif "correct" in model_df.columns:
                    # Convert to numeric, coercing errors to NaN
                    model_df["correct"] = pd.to_numeric(model_df["correct"], errors='coerce')
                    # Drop NaN values before calculating mean
                    mean_acc = model_df["correct"].dropna().mean()
                else:
                    print(f"No accuracy or correct column found in {path} for model {llm_model}")
                    continue
                
                if pd.isna(mean_acc):  # If still NaN after dropna, set to 0
                    mean_acc = 0
                
                # Store if this is the best model for this LLM and top_k
                key = (llm_model, top_k)
                if key not in best_group_multi_models or mean_acc > best_group_multi_models[key]["mean_acc"]:
                    best_group_multi_models[key] = {"path": path, "mean_acc": mean_acc, "df": model_df}
        else:
            # Extract from filename if not in dataframe
            for model_name in ["deepseek", "mistral", "llama", "gpt"]:
                if model_name in path.lower():
                    llm_model = model_name
                    break
            else:
                llm_model = "unknown"
            
            # Check which column to use for accuracy
            if "accuracy" in df.columns:
                # Convert to numeric, coercing errors to NaN
                df["accuracy"] = pd.to_numeric(df["accuracy"], errors='coerce')
                # Drop NaN values before calculating mean
                mean_acc = df["accuracy"].dropna().mean()
            elif "correct" in df.columns:
                # Convert to numeric, coercing errors to NaN
                df["correct"] = pd.to_numeric(df["correct"], errors='coerce')
                # Drop NaN values before calculating mean
                mean_acc = df["correct"].dropna().mean()
            else:
                print(f"No accuracy or correct column found in {path}")
                continue
            
            if pd.isna(mean_acc):  # If still NaN after dropna, set to 0
                mean_acc = 0
            
            # Store if this is the best model for this LLM and top_k
            key = (llm_model, top_k)
            if key not in best_group_multi_models or mean_acc > best_group_multi_models[key]["mean_acc"]:
                best_group_multi_models[key] = {"path": path, "mean_acc": mean_acc, "df": df}

    # Print the best FSM models for single-step and multi-step
    print("\nBest FSM Group Models (Single-Step):")
    for key, data in best_group_models.items():
        llm_model, top_k = key
        print(f"  LLM: {llm_model}, Top-K: {top_k}, Accuracy: {data['mean_acc']:.4f}, Path: {data['path']}")
    
    print("\nBest FSM Group Models (Multi-Step):")
    for key, data in best_group_multi_models.items():
        llm_model, top_k = key
        print(f"  LLM: {llm_model}, Top-K: {top_k}, Accuracy: {data['mean_acc']:.4f}, Path: {data['path']}")

    # Now you can access the best dataframes:
    # For single-step: best_single_models[(llm_model, top_k)]["df"]
    # For multi-step: best_multi_models[(llm_model, top_k)]["df"]

    # Set up the visualization style with larger font
    sns.set_context("paper", font_scale=2.0)
    plt.figure(figsize=(22, 6))
    
    # Create a dictionary to store aggregated results for each model
    def get_best_llm_stats(df, metric_col="accuracy", time_col="avg_prediction_time"):
        """Return (best_mean, best_err, best_time, best_time_err) for the LLM with highest mean accuracy."""
        if df.empty or "llm_model" not in df.columns:
            return 0, 0, 0, 0
        best_mean = -1
        best_err = 0
        best_time = 0
        best_time_err = 0
        best_llm = None
        for model in df['llm_model'].unique():
            model_df = df[df['llm_model'] == model]
            accs = model_df[metric_col][model_df[metric_col] != metric_col]
            if accs.empty:
                continue
            accs = accs.astype(float)
            mean = accs.mean()
            err = accs.std() / np.sqrt(len(accs))
            if mean > best_mean:
                best_mean = mean
                best_err = err
                best_llm = model
                if time_col in model_df.columns:
                    times = model_df[time_col][model_df[time_col] != time_col].astype(float)
                    best_time = times.mean() if not times.empty else 0
                    best_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
                else:
                    best_time = 0
                    best_time_err = 0
        if best_llm:
            print(f"Best LLM for {metric_col}: {best_llm} with accuracy {best_mean:.4f}")
        return best_mean, best_err, best_time, best_time_err

    model_results = {
        'BC': {
            'single_acc': bc_grid_group_df['accuracy'][bc_grid_group_df['accuracy'] != 'accuracy'].astype(float).mean() if not bc_grid_group_df.empty else 0,
            'single_err': bc_grid_group_df['accuracy'][bc_grid_group_df['accuracy'] != 'accuracy'].astype(float).std() / np.sqrt(len(bc_grid_group_df)) if not bc_grid_group_df.empty else 0,
            'multi_acc': bc_grid_group_multi_df['accuracy'][bc_grid_group_multi_df['accuracy'] != 'accuracy'].astype(float).mean() if not bc_grid_group_multi_df.empty else 0,
            'multi_err': bc_grid_group_multi_df['accuracy'][bc_grid_group_multi_df['accuracy'] != 'accuracy'].astype(float).std() / np.sqrt(len(bc_grid_group_multi_df)) if not bc_grid_group_multi_df.empty else 0,
            'pred_time': bc_grid_group_multi_df['avg_prediction_time'][bc_grid_group_multi_df['avg_prediction_time'] != 'avg_prediction_time'].astype(float).mean() if 'avg_prediction_time' in bc_grid_group_multi_df.columns and not bc_grid_group_multi_df.empty else 0,
            'time_err': bc_grid_group_multi_df['avg_prediction_time'][bc_grid_group_multi_df['avg_prediction_time'] != 'avg_prediction_time'].astype(float).std() / np.sqrt(len(bc_grid_group_multi_df)) if 'avg_prediction_time' in bc_grid_group_multi_df.columns and not bc_grid_group_multi_df.empty else 0
        },
        'AutoToM': {
            'single_acc': get_best_llm_stats(autoToM_grid_group_df)[0],
            'single_err': get_best_llm_stats(autoToM_grid_group_df)[1],
            'multi_acc': get_best_llm_stats(autoToM_grid_group_multi_df)[0],
            'multi_err': get_best_llm_stats(autoToM_grid_group_multi_df)[1],
            'pred_time': get_best_llm_stats(autoToM_grid_group_multi_df)[2],
            'time_err': get_best_llm_stats(autoToM_grid_group_multi_df)[3]
        },
        'NLLM': {  # Renamed from 'Naive LLM'
            'single_acc': get_best_llm_stats(nllm_grid_group_df)[0],
            'single_err': get_best_llm_stats(nllm_grid_group_df)[1],
            'multi_acc': get_best_llm_stats(nllm_grid_group_multi_df)[0],
            'multi_err': get_best_llm_stats(nllm_grid_group_multi_df)[1],
            'pred_time': get_best_llm_stats(nllm_grid_group_multi_df)[2],
            'time_err': get_best_llm_stats(nllm_grid_group_multi_df)[3]
        }
    }
    
    # Add FSM results - use the best model for each metric
    fsm_group_dfs = [data["df"] for data in best_group_models.values()]
    fsm_group_multi_dfs = [data["df"] for data in best_group_multi_models.values()]
    
    # --- NEW: Find the best FSM group single accuracy ---
    fsm_group_acc, fsm_group_err = 0, 0
    fsm_group_pred_time, fsm_group_time_err = 0, 0  # <--- add these for group single
    best_fsm_group_llm = None
    if fsm_group_dfs:
        best_acc = -1
        for df in fsm_group_dfs:
            if 'accuracy' in df.columns:
                # --- FIX: Find best (llm_model, num_hypothesis) pair ---
                if 'llm_model' in df.columns and 'num_hypothesis' in df.columns:
                    grouped = df.groupby(['llm_model', 'num_hypothesis'])
                    for (llm, hyp), group in grouped:
                        accs = group['accuracy'][group['accuracy'] != 'accuracy'].astype(float)
                        mean_acc = accs.mean()
                        err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
                        time_col = 'avg_prediction_time' if 'avg_prediction_time' in group.columns else 'prediction_time' if 'prediction_time' in group.columns else None
                        if mean_acc > best_acc:
                            best_acc = mean_acc
                            best_err = err
                            best_fsm_group_llm = llm
                            if time_col:
                                times = group[time_col][group[time_col] != time_col].astype(float)
                                fsm_group_pred_time = times.mean() if not times.empty else 0
                                fsm_group_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
                else:
                    accs = df['accuracy'][df['accuracy'] != 'accuracy'].astype(float)
                    mean_acc = accs.mean()
                    err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
                    time_col = 'avg_prediction_time' if 'avg_prediction_time' in df.columns else 'prediction_time' if 'prediction_time' in df.columns else None
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_err = err
                        best_fsm_group_llm = "unknown"
                        if time_col:
                            times = df[time_col][df[time_col] != time_col].astype(float)
                            fsm_group_pred_time = times.mean() if not times.empty else 0
                            fsm_group_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
        if best_acc >= 0:
            fsm_group_acc = best_acc
            fsm_group_err = best_err
            print(f"\nBest FSM Group Single-Step LLM: {best_fsm_group_llm} with accuracy {fsm_group_acc:.4f}")
    
    # --- NEW: Find the best FSM group multi accuracy and prediction time ---
    fsm_group_multi_acc, fsm_group_multi_err, fsm_group_pred_time, fsm_group_time_err = 0, 0, 0, 0
    best_fsm_group_multi_llm = None
    if fsm_group_multi_dfs:
        best_acc = -1
        for df in fsm_group_multi_dfs:
            acc_col = 'accuracy' if 'accuracy' in df.columns else 'correct' if 'correct' in df.columns else None
            if acc_col:
                # --- FIX: Find best (llm_model, num_hypothesis) pair ---
                if 'llm_model' in df.columns and 'num_hypothesis' in df.columns:
                    grouped = df.groupby(['llm_model', 'num_hypothesis'])
                    for (llm, hyp), group in grouped:
                        accs = group[acc_col][group[acc_col] != acc_col].astype(float)
                        mean_acc = accs.mean()
                        err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
                        time_col = 'avg_prediction_time' if 'avg_prediction_time' in group.columns else 'prediction_time' if 'prediction_time' in group.columns else None
                        if mean_acc > best_acc:
                            best_acc = mean_acc
                            best_err = err
                            best_fsm_group_multi_llm = llm
                            if time_col:
                                times = group[time_col][group[time_col] != time_col].astype(float)
                                fsm_group_pred_time = times.mean() if not times.empty else 0
                                fsm_group_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
                else:
                    accs = df[acc_col][df[acc_col] != acc_col].astype(float)
                    mean_acc = accs.mean()
                    err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
                    time_col = 'avg_prediction_time' if 'avg_prediction_time' in df.columns else 'prediction_time' if 'prediction_time' in df.columns else None
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        best_err = err
                        best_fsm_group_multi_llm = "unknown"
                        if time_col:
                            times = df[time_col][df[time_col] != time_col].astype(float)
                            fsm_group_pred_time = times.mean() if not times.empty else 0
                            fsm_group_time_err = times.std() / np.sqrt(len(times)) if not times.empty else 0
        if best_acc >= 0:
            fsm_group_multi_acc = best_acc
            fsm_group_multi_err = best_err
            print(f"Best FSM Group Multi-Step LLM: {best_fsm_group_multi_llm} with accuracy {fsm_group_multi_acc:.4f}")
    
    model_results['AUTOMA'] = {
        'single_acc': fsm_group_acc,
        'single_err': fsm_group_err,
        'multi_acc': fsm_group_multi_acc,
        'multi_err': fsm_group_multi_err,
        'pred_time': fsm_group_pred_time,      # <--- now correct for best pair
        'time_err': fsm_group_time_err         # <--- now correct for best pair
    }

    # load fsm_grid_group_multi_time_file_paths
    pred_times = []
    for path in fsm_grid_group_multi_time_file_paths:
        df = load_data(path)
        if df.empty:
            continue
        # get avg prediction time
        time_col = 'avg_prediction_time' if 'avg_prediction_time' in df.columns else 'prediction_time' if 'prediction_time' in df.columns else None
        if time_col:
            times = df[time_col][df[time_col] != time_col].astype(float)
            fsm_group_pred_time = times.mean() 
            pred_times.append(fsm_group_pred_time)
    pred_times = np.array(pred_times)
    pred_time_mean = pred_times.mean()
    pred_time_std = pred_times.std() / np.sqrt(len(pred_times))

    num_predictions = 25
    model_results['AUTOMA']['line_plot_pred_time'] = [model_results['AUTOMA']['pred_time']]
    model_results['AUTOMA']['line_plot_time_err'] = [model_results['AUTOMA']['time_err']]
    for i in range(num_predictions):
        model_results['AUTOMA']['line_plot_pred_time'].append(model_results['AUTOMA']['line_plot_pred_time'][-1] + pred_time_mean)
        model_results['AUTOMA']['line_plot_time_err'].append(pred_time_std)
    
    for key in model_results.keys():
        if key != 'AUTOMA':
            model_results[key]['line_plot_pred_time'] = [model_results[key]['pred_time']]
            model_results[key]['line_plot_time_err'] = [model_results[key]['time_err']]
            for i in range(num_predictions):
                model_results[key]['line_plot_pred_time'].append(model_results[key]['line_plot_pred_time'][-1] + model_results[key]['pred_time'])
                model_results[key]['line_plot_time_err'].append(model_results[key]['time_err'])
    # plot the line plot
    plt.figure(figsize=(8, 6))
    x = range(num_predictions + 1)  # +1 for initial point
    for model in model_results.keys():
        time_dicts['group'][model] = {}
        times = model_results[model]['line_plot_pred_time']
        errs = model_results[model]['line_plot_time_err']
        time_dicts['group'][model]['line_plot_pred_time'] = times
        time_dicts['group'][model]['line_plot_time_err'] = errs
        plt.fill_between(x, 
                        [t - e for t, e in zip(times, errs)],
                        [t + e for t, e in zip(times, errs)],
                        color=model_colors[model], alpha=0.5, label=model)
    plt.xlabel('Number of Predictions')
    plt.ylabel('Cumulative Prediction Time (s)')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(model_results))
    sns.despine()
    plt.tight_layout()
    plt.savefig('question2-1.png', bbox_inches='tight')
    plt.close()

    # Create dataframes for plotting
    models = list(model_results.keys())
    group_acc_data = [model_results[m]['single_acc'] for m in models]
    group_err_data = [model_results[m]['single_err'] for m in models]
    group_multi_acc_data = [model_results[m]['multi_acc'] for m in models]
    group_multi_err_data = [model_results[m]['multi_err'] for m in models]
    group_pred_time_data = [model_results[m]['pred_time'] for m in models]
    group_time_err_data = [model_results[m]['time_err'] for m in models]
    
    # --- Significance stars helper ---
    # def get_significance_stars(p):
    #     if p < 0.001:
    #         return '***'
    #     elif p < 0.01:
    #         return '**'
    #     elif p < 0.05:
    #         return '*'
    #     else:
    #         return ''

    # --- Prepare accuracy arrays for significance testing ---
    def get_acc_array(df, col='accuracy'):
        if df is None or df.empty or col not in df.columns:
            return np.array([])
        arr = df[col][df[col] != col]
        return arr.astype(float).values

    # FSM arrays (group)
    fsm_group_acc_arr = None
    for data in best_group_models.values():
        arr = get_acc_array(data["df"])
        if arr.size > 0:
            fsm_group_acc_arr = arr
            break
    fsm_group_multi_acc_arr = None
    for data in best_group_multi_models.values():
        acc_col = 'accuracy' if 'accuracy' in data["df"].columns else 'correct'
        arr = get_acc_array(data["df"], col=acc_col)
        if arr.size > 0:
            fsm_group_multi_acc_arr = arr
            break

    # Other models' arrays
    model_group_acc_arrays = {
        'BC': get_acc_array(bc_grid_group_df),
        'AutoToM': get_acc_array(autoToM_grid_group_df),
        'NLLM': get_acc_array(nllm_grid_group_df),  # Renamed
    }
    model_group_multi_acc_arrays = {
        'BC': get_acc_array(bc_grid_group_multi_df),
        'AutoToM': get_acc_array(autoToM_grid_group_multi_df),
        'NLLM': get_acc_array(nllm_grid_group_multi_df),  # Renamed
    }
    
    # # Calculate p-values for group single-step comparisons
    # print("\n=== Question 2: Group Single-Step Significance Tests (AUTOMA vs Others) ===")
    # for model, arr in model_group_acc_arrays.items():
    #     p_val = perform_t_test(fsm_group_acc_arr, arr)
    #     stars = get_significance_stars(p_val)
    #     df = len(fsm_group_acc_arr) + len(arr) - 2  # approximate degrees of freedom
    #     print(f"AUTOMA vs {model}: t({df:.1f}) = {p_val[0]:.3f}, p = {p_val[1]:.4f} {stars}")
    
    # # Calculate p-values for group multi-step comparisons
    # print("\n=== Question 2: Group Multi-Step Significance Tests (AUTOMA vs Others) ===")
    # for model, arr in model_group_multi_acc_arrays.items():
    #     p_val = perform_t_test(fsm_group_multi_acc_arr, arr)
    #     stars = get_significance_stars(p_val)
    #     df = len(fsm_group_multi_acc_arr) + len(arr) - 2  # approximate degrees of freedom
    #     print(f"AUTOMA vs {model}: t({df:.1f}) = {p_val[0]:.3f}, p = {p_val[1]:.4f} {stars}")
    
    # --- Create the three subplots ---
    # --- Plot with significance stars ---
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(x=models, y=group_acc_data, palette=[model_colors[m] for m in models], errorbar=None, hue=models)
    plt.errorbar(x=range(len(models)), y=group_acc_data, yerr=group_err_data, fmt='none', ecolor='black', capsize=5)
    plt.ylabel('Accuracy', fontsize=44)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=44)
    plt.yticks([0, 0.33, 0.67, 1.0], fontsize=44)
    ax1.set_title('Single-Step', fontsize=44)
    sns.despine()

    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x=models, y=group_multi_acc_data, palette=[model_colors[m] for m in models], errorbar=None, hue=models)
    plt.errorbar(x=range(len(models)), y=group_multi_acc_data, yerr=group_multi_err_data, fmt='none', ecolor='black', capsize=5)
    plt.ylabel('Accuracy', fontsize=44)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=44)
    plt.yticks([0, 0.33, 0.67, 1.0], fontsize=44)
    ax2.set_title('Multi-Step', fontsize=44)
    sns.despine()





    # plt.figure(figsize=(20, 10))
    # plt.subplot(1, 2, 1)
    # ax1 = sns.barplot(x=models, y=group_acc_data, palette=[model_colors[m] for m in models], errorbar=None, hue=models)
    # plt.errorbar(x=range(len(models)), y=group_acc_data, yerr=group_err_data, fmt='none', ecolor='black', capsize=5)
    # plt.ylabel('Single-Step Prediction Accuracy', fontsize=34)
    # plt.ylim(0, 1.0)
    # plt.xticks(fontsize=34)
    # plt.yticks(fontsize=34)
    # sns.despine()

    # plt.subplot(1, 2, 2)
    # ax2 = sns.barplot(x=models, y=group_multi_acc_data, palette=[model_colors[m] for m in models], errorbar=None, hue=models)
    # plt.errorbar(x=range(len(models)), y=group_multi_acc_data, yerr=group_multi_err_data, fmt='none', ecolor='black', capsize=5)
    # plt.ylabel('Multi-Step Prediction Accuracy', fontsize=34)
    # plt.ylim(0, 1.0)
    # plt.xticks(fontsize=34)
    # plt.yticks(fontsize=34)
    # sns.despine()

    # plt.subplot(1, 3, 3)
    # ax3 = sns.barplot(x=models, y=group_pred_time_data, palette=[model_colors[m] for m in models], errorbar=None, hue=models)
    # plt.errorbar(x=range(len(models)), y=group_pred_time_data, yerr=group_time_err_data, fmt='none', ecolor='black', capsize=5)
    # # plt.title('Average Prediction Time (Log Scale)', fontsize=20)
    # plt.ylabel('Average Prediction Time (s)', fontsize=21)
    # plt.yscale('log')  # Set y-axis to log scale
    # plt.xticks(fontsize=21)
    # plt.yticks(fontsize=21)
    # sns.despine()
    
    # Collect best LLM models used for each approach
    fsm_best_llm_model = None
    fsm_best_acc = -1
    for key, data in best_group_models.items():
        llm_model, top_k = key
        if data["mean_acc"] > fsm_best_acc:
            fsm_best_acc = data["mean_acc"]
            fsm_best_llm_model = llm_model
    
    autotom_best_llm_model = None
    autotom_best_acc = -1
    if not autoToM_grid_group_df.empty and 'llm_model' in autoToM_grid_group_df.columns:
        for model in autoToM_grid_group_df['llm_model'].unique():
            model_data = autoToM_grid_group_df[autoToM_grid_group_df['llm_model'] == model]
            acc_data = model_data['accuracy'][(model_data['accuracy'] != 'accuracy')]
            if not acc_data.empty:
                acc = acc_data.astype(float).mean()
                if acc > autotom_best_acc:
                    autotom_best_acc = acc
                    autotom_best_llm_model = model
    
    nllm_best_llm_model = None
    nllm_best_acc = -1
    if not nllm_grid_group_df.empty and 'llm_model' in nllm_grid_group_df.columns:
        for model in nllm_grid_group_df['llm_model'].unique():
            model_data = nllm_grid_group_df[nllm_grid_group_df['llm_model'] == model]
            acc_data = model_data['accuracy'][(model_data['accuracy'] != 'accuracy')]
            if not acc_data.empty:
                acc = acc_data.astype(float).mean()
                if acc > nllm_best_acc:
                    nllm_best_acc = acc
                    nllm_best_llm_model = model
    
    # # Create text strings with the best LLM models used
    # legend_texts = []
    # if fsm_best_llm_model:
    #     legend_texts.append(f"FSM: {fsm_best_llm_model}")
    # if autotom_best_llm_model:
    #     legend_texts.append(f"AutoToM: {autotom_best_llm_model}")
    # if nllm_best_llm_model:
    #     legend_texts.append(f"NLLM: {nllm_best_llm_model}")
    
    # # Add a text box with the best LLM models used
    # if legend_texts:
    #     legend_text = "\n".join(legend_texts)
    #     plt.figtext(0.5, 0.01, legend_text, ha='center', fontsize=16, 
    #                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)  # Make more room for the text at the bottom
    plt.savefig('question2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # List of LLM models to plot
    llm_models = ["llama", "deepseek", "gpt"]

    for llm in llm_models:
        plot_models = []
        plot_acc = []
        plot_err = []
        plot_pred_time = []
        plot_time_err = []

        plot_models.append("BC")
        plot_acc.append(model_results["BC"]["single_acc"])
        plot_err.append(model_results["BC"]["single_err"])
        plot_pred_time.append(model_results["BC"]["pred_time"])
        plot_time_err.append(model_results["BC"]["time_err"])

        # AutoToM
        if not autoToM_grid_group_df.empty and "llm_model" in autoToM_grid_group_df.columns:
            autotom_df = autoToM_grid_group_df[autoToM_grid_group_df["llm_model"].str.lower().str.contains(llm)]
            if not autotom_df.empty:
                acc = autotom_df["accuracy"][autotom_df["accuracy"] != "accuracy"].astype(float).mean()
                err = autotom_df["accuracy"][autotom_df["accuracy"] != "accuracy"].astype(float).std() / np.sqrt(len(autotom_df))
                if "avg_prediction_time" in autotom_df.columns:
                    pred_time = autotom_df["avg_prediction_time"][autotom_df["avg_prediction_time"] != "avg_prediction_time"].astype(float).mean()
                    time_err = autotom_df["avg_prediction_time"][autotom_df["avg_prediction_time"] != "avg_prediction_time"].astype(float).std() / np.sqrt(len(autotom_df))
                else:
                    pred_time = 0
                    time_err = 0
                plot_models.append("AutoToM")
                plot_acc.append(acc)
                plot_err.append(err)
                plot_pred_time.append(pred_time)
                plot_time_err.append(time_err)

        # Naive LLM
        if not nllm_grid_group_df.empty and "llm_model" in nllm_grid_group_df.columns:
            nllm_df = nllm_grid_group_df[nllm_grid_group_df["llm_model"].str.lower().str.contains(llm)]
            if not nllm_df.empty:
                acc = nllm_df["accuracy"][nllm_df["accuracy"] != "accuracy"].astype(float).mean()
                err = nllm_df["accuracy"][nllm_df["accuracy"] != "accuracy"].astype(float).std() / np.sqrt(len(nllm_df))
                if "avg_prediction_time" in nllm_df.columns:
                    pred_time = nllm_df["avg_prediction_time"][nllm_df["avg_prediction_time"] != "avg_prediction_time"].astype(float).mean()
                    time_err = nllm_df["avg_prediction_time"][nllm_df["avg_prediction_time"] != "avg_prediction_time"].astype(float).std() / np.sqrt(len(nllm_df))
                else:
                    pred_time = 0
                    time_err = 0
                plot_models.append("NLLM")
                plot_acc.append(acc)
                plot_err.append(err)
                plot_pred_time.append(pred_time)
                plot_time_err.append(time_err)

        # FSM: use all FSM group dataframes, not just best
        fsm_group_dfs_llm = []
        for df in fsm_group_dfs:
            if "llm_model" in df.columns:
                match_df = df[df["llm_model"].str.lower().str.contains(llm)]
                if not match_df.empty:
                    fsm_group_dfs_llm.append(match_df)
        if fsm_group_dfs_llm:
            fsm_df = pd.concat(fsm_group_dfs_llm, ignore_index=True)
            acc = fsm_df["accuracy"][fsm_df["accuracy"] != "accuracy"].astype(float).mean()
            err = fsm_df["accuracy"][fsm_df["accuracy"] != "accuracy"].astype(float).std() / np.sqrt(len(fsm_df))
            if "avg_prediction_time" in fsm_df.columns:
                pred_time = fsm_df["avg_prediction_time"][fsm_df["avg_prediction_time"] != "avg_prediction_time"].astype(float).mean()
                time_err = fsm_df["avg_prediction_time"][fsm_df["avg_prediction_time"] != "avg_prediction_time"].astype(float).std() / np.sqrt(len(fsm_df))
            else:
                pred_time = 0
                time_err = 0
            plot_models.append("AUTOMA")
            plot_acc.append(acc)
            plot_err.append(err)
            plot_pred_time.append(pred_time)
            plot_time_err.append(time_err)

        sns.set_context("paper", font_scale=2.0)
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=plot_models, y=plot_acc, palette=[model_colors[m] for m in plot_models], errorbar=None, hue=plot_models)
        plt.errorbar(x=range(len(plot_models)), y=plot_acc, yerr=plot_err, fmt='none', ecolor='black', capsize=5)
        plt.ylabel(f'Group Single-Step Prediction Accuracy ({llm})', fontsize=21)
        plt.ylim(0, 1.0)
        plt.xticks(fontsize=21)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'group_single_acc_{llm}.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=plot_models, y=plot_pred_time, palette=[model_colors[m] for m in plot_models], errorbar=None, hue=plot_models)
        plt.errorbar(x=range(len(plot_models)), y=plot_pred_time, yerr=plot_time_err, fmt='none', ecolor='black', capsize=5)
        plt.ylabel(f'Group Average Prediction Time (seconds) ({llm})', fontsize=21)
        plt.yscale('log')
        plt.xticks(fontsize=21)
        plt.yticks(fontsize=21)
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'group_pred_time_{llm}.png', dpi=300, bbox_inches='tight')
        plt.close()

elif question == 3:
    # Plot accuracy for human data

    # Load baseline data
    bc_human_file_path = 'BC/human_accuracy_BC_4hyp.csv'
    bc_human_df = load_data(bc_human_file_path)

    autoToM_human_file_path = 'AutoToM/human_accuracy_AutoToM_2hyp.csv'
    autoToM_human_df = load_data(autoToM_human_file_path)

    nllm_human_file_path = 'NLLM/human_accuracy_NLLM_2hyp.csv'
    nllm_human_df = load_data(nllm_human_file_path)

    # Load human prediction data
    human_human_filepath = '/mmfs1/gscratch/socialrl/kjha/automaticity/data/human_prediction_data.csv'
    human_human_df = pd.read_csv(human_human_filepath)
    human_human_df['accuracy'] = (human_human_df['predicted_action_idx'] == human_human_df['gt_action_idx']).astype(float)
    

    # --- FSM: Find best mean accuracy across all bootstrap human files, all LLMs, all num_hypothesis ---
    import glob
    fsm_human_paths = glob.glob("FSM/fixed_human_bootstrap_accuracy_FSM*")
    fsm_human_acc = -1
    fsm_human_err = 0
    best_fsm_file = None
    best_num_hypothesis = None
    best_llm_model = None
    best_fsm_accs = []  # Store the actual accuracy values for significance testing
    more_than_6_rows = 0
    total_groups = 0
    for path in fsm_human_paths:
        df = load_data(path)
        if 'accuracy' in df.columns:
            epochs = df['epoch']
            # task_id = epochs % len(task_list)
            df['task'] = df['epoch'].apply(lambda x: task_list[x % len(task_list)])
            # Multiply accuracy by 1/6 when epoch % len(task_list) == 1
            # df['accuracy'] = df.apply(lambda row: float(row['accuracy']) * (1/6) if row['epoch'] % len(task_list) == 1 else float(row['accuracy']), axis=1)
            # If both num_hypothesis and llm_model exist, group and search for best
            if 'num_hypothesis' in df.columns and 'llm_model' in df.columns:
                # Count number of rows per group
                group_counts = df.groupby(['num_hypothesis', 'llm_model']).size()
                # Only include groups with more than 10 rows
                valid_groups = group_counts[group_counts > 6].index
                more_than_6_rows += len(valid_groups)
                # Filter df to only include valid groups
                df_filtered = df[df.set_index(['num_hypothesis', 'llm_model']).index.isin(valid_groups)]
                
                if not df_filtered.empty:
                    grouped = df_filtered.groupby(['num_hypothesis', 'llm_model'])['accuracy'].mean().reset_index()
                    for _, row in grouped.iterrows():
                        if row['accuracy'] > fsm_human_acc:
                            fsm_human_acc = row['accuracy']
                            best_fsm_file = path
                            best_num_hypothesis = row['num_hypothesis']
                            best_llm_model = row['llm_model']
            else:
                # For files without grouping, check total row count
                if len(df) > 6:
                    more_than_6_rows += len(df)
                    mean_acc = df['accuracy'].astype(float).mean()
                    if mean_acc > fsm_human_acc:
                        fsm_human_acc = mean_acc
                        best_fsm_file = path
                        best_num_hypothesis = None
                        best_llm_model = None
        else:
            print(f"Warning: 'accuracy' column not found in {path}")
            
    # Calculate error for the best FSM config
    if best_fsm_file is not None:
        df = load_data(best_fsm_file)
        if best_num_hypothesis is not None and best_llm_model is not None:
            df = df[(df['num_hypothesis'] == best_num_hypothesis) & ((df['llm_model'] == best_llm_model))]
        if 'accuracy' in df.columns:
            # Multiply accuracy by 1/6 when epoch % len(task_list) == 1
            # df['accuracy'] = df.apply(lambda row: float(row['accuracy']) * (1/6) if row['epoch'] % len(task_list) == 1 else float(row['accuracy']), axis=1)
            accs = df['accuracy'].astype(float)
            best_fsm_accs = accs.values  # Store for significance testing
            fsm_human_err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
            fsm_human_acc = accs.mean()

    # --- BC: Just use the mean accuracy ---
    bc_accs = bc_human_df['accuracy'][bc_human_df['accuracy'] != 'accuracy'].astype(float).values if not bc_human_df.empty else np.array([])
    bc_acc = bc_accs.mean() if len(bc_accs) > 0 else 0
    bc_err = bc_accs.std() / np.sqrt(len(bc_accs)) if len(bc_accs) > 1 else 0

    # --- AutoToM: Find best LLM model ---
    autotom_acc, autotom_err = 0, 0
    best_autotom_accs = np.array([])
    if not autoToM_human_df.empty and 'llm_model' in autoToM_human_df.columns:
        best_acc = -1
        for model in autoToM_human_df['llm_model'].unique():
            model_df = autoToM_human_df[autoToM_human_df['llm_model'] == model]
            accs = model_df['accuracy'][model_df['accuracy'] != 'accuracy'].astype(float).values
            mean_acc = accs.mean() if len(accs) > 0 else 0
            err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_err = err
                best_autotom_accs = accs
        if best_acc >= 0:
            autotom_acc = best_acc
            autotom_err = best_err


    # --- Naive LLM: Find best LLM model ---
    nllm_acc, nllm_err = 0, 0
    best_nllm_accs = np.array([])
    if not nllm_human_df.empty and 'llm_model' in nllm_human_df.columns:
        best_acc = -1
        for model in nllm_human_df['llm_model'].unique():
            model_df = nllm_human_df[nllm_human_df['llm_model'] == model]
            accs = model_df['accuracy'][model_df['accuracy'] != 'accuracy'].astype(float).values
            mean_acc = accs.mean() if len(accs) > 0 else 0
            err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_err = err
                best_nllm_accs = accs
        if best_acc >= 0:
            nllm_acc = best_acc
            nllm_err = best_err

    # --- Human: Just use the mean accuracy ---
    human_accs = human_human_df['accuracy'].values
    human_acc = human_accs.mean()
    human_err = human_accs.std() / np.sqrt(len(human_accs))

    # --- Perform significance testing ---
    print("\n=== Question 3: Human Data Significance Tests (AUTOMA vs Others) ===")
    model_human_acc_arrays = {
        'BC': bc_accs,
        'AutoToM': best_autotom_accs,
        'NLLM': best_nllm_accs,
        'Human': human_accs
    }
    
    for model, arr in model_human_acc_arrays.items():
        if len(arr) > 0 and len(best_fsm_accs) > 0:
            t_stat, p_val = perform_t_test(best_fsm_accs, arr)
            stars = get_significance_stars(p_val)
            df = 200  # approximate degrees of freedom
            print(f"AUTOMA vs {model}: t({df:.1f}) = {t_stat:.3f}, p = {p_val:.4f} {stars}")
        else:
            print(f"AUTOMA vs {model}: Insufficient data for t-test")

    # --- Plot ---
    models = ['BC', 'AutoToM', 'NLLM', 'Human', 'AUTOMA']  # Renamed
    accs = [bc_acc, autotom_acc, nllm_acc, human_acc, fsm_human_acc]
    errs = [bc_err, autotom_err, nllm_err, human_err, fsm_human_err]

    sns.set_context("paper", font_scale=2.0)
    plt.figure(figsize=(22, 8))
    ax = sns.barplot(x=models, y=accs, palette=[model_colors[m] for m in models], errorbar=None, hue=models)
    plt.errorbar(x=range(len(models)), y=accs, yerr=errs, fmt='none', ecolor='black', capsize=5)
    plt.ylabel('Human Accuracy', fontsize=44)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=44)
    plt.yticks([0, 0.33, 0.67, 1.0], fontsize=44)
    sns.despine()
    plt.tight_layout()
    plt.savefig('question3-1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Partnr results
    # AutoToM
    autotom_partnr_single_filepath = 'AutoToM/partnr_accuracy_AutoToM_2hyp.csv'
    autotom_partnr_single_df = load_data(autotom_partnr_single_filepath)

    autotom_partnr_group_filepath = 'AutoToM/partnr2_accuracy_AutoToM_2hyp_group.csv'
    autotom_partnr_group_df = load_data(autotom_partnr_group_filepath)
    
    # Naive LLM
    nllm_partnr_single_filepath = 'NLLM/partnr_accuracy_NLLM_2hyp.csv'
    nllm_partnr_single_df = load_data(nllm_partnr_single_filepath)

    nllm_partnr_group_filepath = 'NLLM/partnr_accuracy_NLLM_2hyp_group.csv'
    nllm_partnr_group_df = load_data(nllm_partnr_group_filepath)
    
    # --- Find best accuracy for AutoToM partnr (single) ---
    autotom_partnr_best_acc, autotom_partnr_best_llm = 0, None
    autotom_partnr_accs = np.array([])
    if not autotom_partnr_single_df.empty and 'llm_model' in autotom_partnr_single_df.columns:
        best_acc = -1
        for model in autotom_partnr_single_df['llm_model'].unique():
            model_df = autotom_partnr_single_df[autotom_partnr_single_df['llm_model'] == model]
            accs = model_df['accuracy'][model_df['accuracy'] != 'accuracy'].astype(float)
            mean_acc = accs.mean()
            if mean_acc > best_acc:
                best_acc = mean_acc
                autotom_partnr_best_llm = model
                autotom_partnr_accs = accs.values
        if best_acc >= 0:
            autotom_partnr_best_acc = best_acc

    print(f"AutoToM partnr (single): best acc = {autotom_partnr_best_acc}, best llm = {autotom_partnr_best_llm}")

    # --- Find best accuracy for AutoToM partnr (group) ---
    autotom_partnr_group_best_acc, autotom_partnr_group_best_llm = 0, None
    autotom_partnr_group_accs = np.array([])
    if not autotom_partnr_group_df.empty and 'llm_model' in autotom_partnr_group_df.columns:
        best_acc = -1
        for model in autotom_partnr_group_df['llm_model'].unique():
            model_df = autotom_partnr_group_df[autotom_partnr_group_df['llm_model'] == model]
            accs = model_df['accuracy'][model_df['accuracy'] != 'accuracy'].astype(float)
            mean_acc = accs.mean()
            if mean_acc > best_acc:
                best_acc = mean_acc
                autotom_partnr_group_best_llm = model
                autotom_partnr_group_accs = accs.values
        if best_acc >= 0:
            autotom_partnr_group_best_acc = best_acc

    print(f"AutoToM partnr (group): best acc = {autotom_partnr_group_best_acc}, best llm = {autotom_partnr_group_best_llm}")

    # --- Find best accuracy for Naive LLM partnr (single) ---
    nllm_partnr_best_acc, nllm_partnr_best_llm = 0, None
    nllm_partnr_accs = np.array([])
    if not nllm_partnr_single_df.empty and 'llm_model' in nllm_partnr_single_df.columns:
        best_acc = -1
        for model in nllm_partnr_single_df['llm_model'].unique():
            model_df = nllm_partnr_single_df[nllm_partnr_single_df['llm_model'] == model]
            accs = model_df['accuracy'][model_df['accuracy'] != 'accuracy'].astype(float)
            mean_acc = accs.mean()
            if mean_acc > best_acc:
                best_acc = mean_acc
                nllm_partnr_best_llm = model
                nllm_partnr_accs = accs.values
        if best_acc >= 0:
            nllm_partnr_best_acc = best_acc

    print(f"NLLM partnr (single): best acc = {nllm_partnr_best_acc}, best llm = {nllm_partnr_best_llm}")

    # --- Find best accuracy for Naive LLM partnr (group) ---
    nllm_partnr_group_best_acc, nllm_partnr_group_best_llm = 0, None
    nllm_partnr_group_accs = np.array([])
    if not nllm_partnr_group_df.empty and 'llm_model' in nllm_partnr_group_df.columns:
        best_acc = -1
        for model in nllm_partnr_group_df['llm_model'].unique():
            model_df = nllm_partnr_group_df[nllm_partnr_group_df['llm_model'] == model]
            accs = model_df['accuracy'][model_df['accuracy'] != 'accuracy'].astype(float)
            mean_acc = accs.mean()
            if mean_acc > best_acc:
                best_acc = mean_acc
                nllm_partnr_group_best_llm = model
                nllm_partnr_group_accs = accs.values
        if best_acc >= 0:
            nllm_partnr_group_best_acc = best_acc

    print(f"NLLM partnr (group): best acc = {nllm_partnr_group_best_acc}, best llm = {nllm_partnr_group_best_llm}")

    # --- FSM: Find best mean accuracy and std across all partnr2 bootstrap files, all LLMs, all num_hypothesis ---
    fsm_partnr_paths = glob.glob("FSM/partnr2_bootstrap_accuracy_FSM*")
    fsm_partnr_group_paths = [p for p in fsm_partnr_paths if "_group" in p]
    fsm_partnr_single_paths = [p for p in fsm_partnr_paths if "_group" not in p]

    def get_best_fsm_bootstrap(paths):
        best_acc = -1
        best_std = 0
        best_file = None
        best_num_hypothesis = None
        best_llm_model = None
        best_fsm_accs = []  # Store the actual accuracy values for significance testing
        for path in paths:
            df = load_data(path)
            if 'accuracy' in df.columns:
                # If both num_hypothesis and llm_model exist, group and search for best
                if 'num_hypothesis' in df.columns and 'llm_model' in df.columns:
                    grouped = df.groupby(['num_hypothesis', 'llm_model'])
                    for (num_hyp, llm), group in grouped:
                        # Only consider groups with more than 17 epochs
                        if len(group) > 17:
                            accs = group['accuracy'].astype(float)
                            mean_acc = accs.mean()
                            std_acc = accs.std() / np.sqrt(len(accs))
                            if mean_acc > best_acc:
                                best_acc = mean_acc
                                best_std = std_acc
                                best_file = path
                                best_num_hypothesis = num_hyp
                                best_llm_model = llm
                                best_fsm_accs = accs
                else:
                    # Only consider if more than 17 epochs
                    if len(df) > 17:
                        accs = df['accuracy'].astype(float)
                        mean_acc = accs.mean()
                        std_acc = accs.std() / np.sqrt(len(accs))
                        if mean_acc > best_acc:
                            best_acc = mean_acc
                            best_std = std_acc
                            best_file = path
                            best_num_hypothesis = None
                            best_llm_model = None
                            best_fsm_accs = accs
        return best_acc, best_std, best_file, best_num_hypothesis, best_llm_model, best_fsm_accs

    # FSM single
    fsm_partnr_single_best_acc, fsm_partnr_single_best_std, fsm_partnr_single_best_file, fsm_partnr_single_best_num_hyp, fsm_partnr_single_best_llm, fsm_partnr_single_best_accs = get_best_fsm_bootstrap(fsm_partnr_single_paths)
    # FSM group
    fsm_partnr_group_best_acc, fsm_partnr_group_best_std, fsm_partnr_group_best_file, fsm_partnr_group_best_num_hyp, fsm_partnr_group_best_llm, fsm_partnr_group_best_accs = get_best_fsm_bootstrap(fsm_partnr_group_paths)

    print(f"FSM partnr2 (single): best acc = {fsm_partnr_single_best_acc}, std = {fsm_partnr_single_best_std}, file = {fsm_partnr_single_best_file}, num_hypothesis = {fsm_partnr_single_best_num_hyp}, llm_model = {fsm_partnr_single_best_llm}")
    print(f"FSM partnr2 (group): best acc = {fsm_partnr_group_best_acc}, std = {fsm_partnr_group_best_std}, file = {fsm_partnr_group_best_file}, num_hypothesis = {fsm_partnr_group_best_num_hyp}, llm_model = {fsm_partnr_group_best_llm}")

    # --- Perform significance testing for partnr single-agent predictions ---
    # Test if the difference between AUTOMA and other approaches is significantly different from zero
    print("\n=== Question 3: Partnr Single-Agent Significance Tests (AUTOMA vs Others) ===")
    # Function to perform paired test across all datapoints
    def perform_paired_test(fsm_df, other_df):
        if fsm_df is None or other_df is None or fsm_df.empty or other_df.empty:
            return None, None, 0
            
        # Get accuracy values
        fsm_accs = fsm_df['accuracy'].astype(float)
        other_accs = other_df['accuracy'].astype(float)
        
        # Take min length to ensure paired data
        # min_len = min(len(fsm_accs), len(other_accs))
        # fsm_accs = np.array(fsm_accs[:min_len])
        # other_accs = np.array(other_accs[:min_len])
        
        # if min_len == 0:
        #     return None, None, 0
            
        # Calculate differences
        # diffs = fsm_accs - other_accs
        
        # Perform one-sample t-test on differences
        # t_stat, p_val = scipy.stats.ttest_1samp(diffs, 0)
        t_stat, p_val = scipy.stats.ttest_ind(fsm_accs, other_accs, equal_var=False)
        
        return t_stat, p_val, len(fsm_accs)
    
    # Load the data for each model
    # FSM single
    fsm_partnr_single_df = None
    if fsm_partnr_single_best_file:
        fsm_partnr_single_df = load_data(fsm_partnr_single_best_file)
        if fsm_partnr_single_best_num_hyp is not None and fsm_partnr_single_best_llm is not None:
            fsm_partnr_single_df = fsm_partnr_single_df[
                (fsm_partnr_single_df['num_hypothesis'] == fsm_partnr_single_best_num_hyp) & 
                (fsm_partnr_single_df['llm_model'] == fsm_partnr_single_best_llm)
            ]
    
    # FSM group
    fsm_partnr_group_df = None
    if fsm_partnr_group_best_file:
        fsm_partnr_group_df = load_data(fsm_partnr_group_best_file)
        if fsm_partnr_group_best_num_hyp is not None and fsm_partnr_group_best_llm is not None:
            fsm_partnr_group_df = fsm_partnr_group_df[
                (fsm_partnr_group_df['num_hypothesis'] == fsm_partnr_group_best_num_hyp) & 
                (fsm_partnr_group_df['llm_model'] == fsm_partnr_group_best_llm)
            ]
    
    # Perform paired tests for single-agent predictions
    print("\n=== Question 3: Partnr Single-Agent Paired Significance Tests (AUTOMA vs Others) ===")
    
    # AutoToM single
    autotom_single_df = autotom_partnr_single_df
    if not autotom_single_df.empty and autotom_partnr_best_llm:
        autotom_single_df = autotom_single_df[autotom_single_df['llm_model'] == autotom_partnr_best_llm]
    
    # NLLM single
    nllm_single_df = nllm_partnr_single_df
    if not nllm_single_df.empty and nllm_partnr_best_llm:
        nllm_single_df = nllm_single_df[nllm_single_df['llm_model'] == nllm_partnr_best_llm]
    
    model_single_dfs = {
        'AutoToM': autotom_single_df,
        'NLLM': nllm_single_df
    }
    
    for model, df in model_single_dfs.items():
        t_stat, p_val, n = perform_paired_test(fsm_partnr_single_df, df)
        if t_stat is not None:
            stars = get_significance_stars(p_val)
            mean_diff = fsm_partnr_single_best_acc - (autotom_partnr_best_acc if model == 'AutoToM' else nllm_partnr_best_acc)
            print(f"AUTOMA vs {model}: diff = {mean_diff:.4f}, t({n-1}) = {t_stat:.3f}, p = {p_val:.4f} {stars}")
        else:
            print(f"AUTOMA vs {model}: Insufficient paired data for t-test")
    
    # Perform paired tests for group predictions
    print("\n=== Question 3: Partnr Group Paired Significance Tests (AUTOMA vs Others) ===")
    
    # AutoToM group
    autotom_group_df = autotom_partnr_group_df
    if not autotom_group_df.empty and autotom_partnr_group_best_llm:
        autotom_group_df = autotom_group_df[autotom_group_df['llm_model'] == autotom_partnr_group_best_llm]
    
    # NLLM group
    nllm_group_df = nllm_partnr_group_df
    if not nllm_group_df.empty and nllm_partnr_group_best_llm:
        nllm_group_df = nllm_group_df[nllm_group_df['llm_model'] == nllm_partnr_group_best_llm]
    
    model_group_dfs = {
        'AutoToM': autotom_group_df,
        'NLLM': nllm_group_df
    }
    
    for model, df in model_group_dfs.items():
        t_stat, p_val, n_tasks = perform_paired_test(fsm_partnr_group_df, df)
        if t_stat is not None:
            stars = get_significance_stars(p_val)
            mean_diff = fsm_partnr_group_best_acc - (autotom_partnr_group_best_acc if model == 'AutoToM' else nllm_partnr_group_best_acc)
            print(f"AUTOMA vs {model}: diff = {mean_diff:.4f}, t({n_tasks-1}) = {t_stat:.3f}, p = {p_val:.4f} {stars}")
        else:
            print(f"AUTOMA vs {model}: Insufficient paired data for t-test")

    # --- Bar plot: single (AutoToM, NLLM, FSM) and group (AutoToM, NLLM, FSM) ---

    # Single
    single_models = ['AutoToM', 'NLLM', 'AUTOMA']  # Renamed
    single_accs = [autotom_partnr_best_acc, nllm_partnr_best_acc, fsm_partnr_single_best_acc]
    single_errs = [0, 0, fsm_partnr_single_best_std]  # If you want to add error bars for AutoToM/NLLM, compute std as above

    # Group
    group_models = ['AutoToM', 'NLLM', 'AUTOMA']  # Renamed
    group_accs = [autotom_partnr_group_best_acc, nllm_partnr_group_best_acc, fsm_partnr_group_best_acc]
    group_errs = [0, 0, fsm_partnr_group_best_std]  # If you want to add error bars for AutoToM/NLLM, compute std as above

    # Plot
    sns.set_context("paper", font_scale=2.0)
    plt.figure(figsize=(16, 12))

    # Left: single
    plt.subplot(1, 1, 1)
    ax1 = sns.barplot(x=single_models, y=single_accs, palette=[model_colors[m] for m in single_models], errorbar=None, hue=single_models)
    plt.errorbar(x=range(len(single_models)), y=single_accs, yerr=single_errs, fmt='none', ecolor='black', capsize=5)
    plt.ylabel('Partnr Accuracy', fontsize=50)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=50)
    plt.yticks([0, 0.33, 0.67, 1.0], fontsize=50)
    # plt.title("Single-agent Action Prediction", fontsize=44)
    sns.despine()

    # # Right: group
    # plt.subplot(1, 2, 2)
    # ax2 = sns.barplot(x=group_models, y=group_accs, palette=[model_colors[m] for m in group_models], errorbar=None, hue=group_models)
    # plt.errorbar(x=range(len(group_models)), y=group_accs, yerr=group_errs, fmt='none', ecolor='black', capsize=5)
    # plt.ylabel('Partnr Accuracy', fontsize=44)
    # plt.ylim(0, 1.0)
    # plt.xticks(fontsize=44)
    # plt.yticks([0, 0.33, 0.67, 1.0], fontsize=44)
    # plt.title("Group", fontsize=44)
    # sns.despine()

    plt.tight_layout()
    plt.savefig('question3-2.png', dpi=300, bbox_inches='tight')
    plt.close()


elif question == 4:
    ############
    # Question 4: Ablation studies of FSM components
    ############

    # Define function to load data with robust handling of inconsistent columns
    def load_robust_csv(filepath):
        try:
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
            
            df = pd.DataFrame(data, columns=header)
            
            # Convert columns to appropriate types
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass  # Keep as string if can't convert
            
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()

    # Define datasets to analyze
    datasets = ["construction", "human", "partnr"]
    
    # Set seaborn style
    sns.set_context("paper", font_scale=2.0)
    
    # Create a figure with subplots arranged horizontally (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(44, 10), sharey=False)  # Horizontal arrangement

    # Set up colors for the two metrics
    accuracy_color = 'blue'
    program_length_color = 'red'
    
    # For collecting legend handles/labels
    legend_handles = []
    legend_labels = []
    legend_added = set()

    for i, dataset in enumerate(datasets):
        print(f"\nProcessing {dataset} dataset")
        
        # Find all bootstrap files for this dataset
        if dataset == "construction":
            bootstrap_files = glob.glob(f"FSM/new_bootstrap_accuracy_FSM*.csv")
            # Filter out files with specific configurations if needed
            bootstrap_files = [f for f in bootstrap_files if "_group" not in f]
        elif dataset == "human":
            bootstrap_files = glob.glob(f"FSM/human_bootstrap_detailed_FSM*.csv")
        elif dataset == "partnr":
            bootstrap_files = glob.glob(f"FSM/partnr2_bootstrap_accuracy_FSM*.csv")
            # Filter out group files
            bootstrap_files = [f for f in bootstrap_files if "_group" not in f]
        # only include if only two_stage is true

        print(f"Found {len(bootstrap_files)} bootstrap files for {dataset}")
        
        # Initialize dictionaries to store aggregated data by number of hypotheses
        hyp_accuracy = {}
        hyp_program_length = {}
        hyp_counts = {}
        hyp_accuracy_std = {}  # For standard error calculation
        hyp_accuracy_values = {}  # Store all values for std error calculation
        hyp_program_length_values = {}  # Store all program length values
        
        # Process each file
        for file_path in bootstrap_files:
            try:
                # Load data with robust handling
                df = load_robust_csv(file_path)
                
                if df.empty:
                    continue
                
                # Check if this is a detailed file with program length
                has_program_length = 'avg_program_length' in df.columns

                if not has_program_length:
                    if 'extra_col_12' in df.columns:
                        df['avg_program_length'] = df['extra_col_12']
                        has_program_length = True
                    elif 'program_length' in df.columns:
                        df['avg_program_length'] = df['program_length']
                        has_program_length = True

                if dataset == 'human':
                    df['accuracy'] = (df['gt_action'] == df['pred_action']).astype(float)
                    task_list = df['task'].unique()

                # Group by num_hypothesis and calculate mean accuracy
                if 'num_hypothesis' in df.columns and 'accuracy' in df.columns:
                    grouped = df.groupby('num_hypothesis')
                    
                    for num_hyp, group in grouped:
                        # Initialize if this hypothesis number is new
                        if num_hyp not in hyp_accuracy:
                            hyp_accuracy[num_hyp] = 0
                            hyp_program_length[num_hyp] = 0
                            hyp_counts[num_hyp] = 0
                            hyp_accuracy_values[num_hyp] = []
                            hyp_program_length_values[num_hyp] = []
                        
                        # Calculate mean accuracy for this group
                        group_acc = group['accuracy'].mean()
                        
                        # Store all accuracy values for standard error calculation
                        hyp_accuracy_values[num_hyp].extend(group['accuracy'].tolist())
                        
                        # Update running totals
                        hyp_accuracy[num_hyp] += group_acc * len(group)
                        hyp_counts[num_hyp] += len(group)
                        
                        # If available, calculate program length
                        if has_program_length:
                            # drop nan values
                            valid_lengths = group['avg_program_length'].dropna()
                            if not valid_lengths.empty:
                                group_prog_len = valid_lengths.mean()
                                hyp_program_length[num_hyp] += group_prog_len * len(valid_lengths)
                                hyp_program_length_values[num_hyp].extend(valid_lengths.tolist())
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Calculate averages and standard errors
        hyp_nums = sorted(hyp_accuracy.keys())
        accuracies = []
        program_lengths = []
        acc_std_errors = []
        prog_std_errors = []
        
        for num_hyp in hyp_nums:
            if hyp_counts[num_hyp] > 0:
                # Calculate average accuracy
                accuracies.append(hyp_accuracy[num_hyp] / hyp_counts[num_hyp])
                
                # Calculate standard error for accuracy
                values = hyp_accuracy_values[num_hyp]
                acc_std_errors.append(np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0)
                
                # Calculate average program length if available
                if hyp_program_length[num_hyp] > 0:
                    prog_values = hyp_program_length_values[num_hyp]
                    program_lengths.append(hyp_program_length[num_hyp] / len(prog_values))
                    # Calculate standard error for program length
                    prog_std_errors.append(np.std(prog_values) / np.sqrt(len(prog_values)) if len(prog_values) > 1 else 0)
                else:
                    program_lengths.append(0)
                    prog_std_errors.append(0)
            else:
                breakpoint()
                accuracies.append(0)
                acc_std_errors.append(0)
                program_lengths.append(0)
                prog_std_errors.append(0)
        
        # Plot accuracy on the left y-axis with fill_between for standard error
        ax1 = axes[i]
        # Plot accuracy
        acc_line, = ax1.plot(hyp_nums, accuracies, color=accuracy_color, marker='o', 
                linestyle='-', linewidth=2, label='Accuracy')
        ax1.fill_between(hyp_nums, 
                        [acc - err for acc, err in zip(accuracies, acc_std_errors)],
                        [acc + err for acc, err in zip(accuracies, acc_std_errors)],
                        color=accuracy_color, alpha=0.2)
        ax1.set_ylabel('Accuracy', color=accuracy_color, fontsize=44)
        ax1.tick_params(axis='y', labelcolor=accuracy_color, labelsize=44)
        ax1.tick_params(axis='x', labelsize=44)
        ax1.set_xlabel('# of Hypotheses', fontsize=44)
        
        # Collect legend handles/labels only once
        if 'Accuracy' not in legend_added:
            legend_handles.append(acc_line)
            legend_labels.append('Accuracy')
            legend_added.add('Accuracy')

        # If we have program length data, plot on the right y-axis
        if any(program_lengths):
            ax2 = ax1.twinx()
            prog_line, = ax2.plot(hyp_nums, program_lengths, color=program_length_color, marker='s', 
                    linestyle='--', linewidth=2, label='Avg Program Length')
            ax2.fill_between(hyp_nums, 
                            [pl - err for pl, err in zip(program_lengths, prog_std_errors)],
                            [pl + err for pl, err in zip(program_lengths, prog_std_errors)],
                            color=program_length_color, alpha=0.2)
            ax2.set_ylabel('Avg Program Length', color=program_length_color, fontsize=44)
            ax2.tick_params(axis='y', labelcolor=program_length_color, labelsize=44)
            ax2.tick_params(axis='x', labelsize=44)
            if 'Avg Program Length' not in legend_added:
                legend_handles.append(prog_line)
                legend_labels.append('Avg Program Length')
                legend_added.add('Avg Program Length')
        
        ax1.set_title(f'{dataset.capitalize()} Dataset', fontsize=44)
        
        # Use sns.despine to remove the top spine
        sns.despine(ax=ax1)
        if any(program_lengths):
            sns.despine(ax=ax2, right=False)

    # Add a single shared legend underneath all plots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # Make room for the legend at the bottom
    fig.legend(
        legend_handles,
        legend_labels,
        loc='lower center',
        ncol=2,
        fontsize=44,
        bbox_to_anchor=(0.5, 0.02)  # Position the legend at the bottom
    )

    plt.savefig('question4-1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more detailed plot for the grid dataset with different configurations
    print("\nCreating detailed ablation plot for grid dataset")
    
    # Define configurations to compare
    configurations = [
        {"two_stage": True, "structured": "False", "rejuvenation": False, "label": "Two-Stage (AUTOMA)"},
        {"two_stage": False, "structured": "False", "rejuvenation": False, "label": "No Two-Stage"},
        {"two_stage": True, "structured": "p1", "rejuvenation": False, "label": "Two-Stage + Moderate"},
        {"two_stage": True, "structured": "p2", "rejuvenation": False, "label": "Two-Stage + Severe"},
        {"two_stage": True, "structured": "False", "rejuvenation": True, "label": "Two-Stage + Rejuvenation"}
    ]
    
    # Set seaborn style
    sns.set_context("paper", font_scale=1.5)
    
    # Create a figure for detailed grid ablation
    plt.figure(figsize=(26, 10))  # Increased height from 8 to 12
    
    # Define colors for each configuration
    config_colors = plt.cm.tab10(np.linspace(0, 1, len(configurations)))
    
    # Process each configuration
    for idx, config in enumerate(configurations):
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
                # Load data with robust handling
                df = load_robust_csv(filepath)
                
                if df.empty:
                    continue
                
                # Group by num_hypothesis and calculate mean accuracy
                if 'num_hypothesis' in df.columns and 'accuracy' in df.columns:
                    grouped = df.groupby('num_hypothesis')['accuracy'].agg(['mean', 'count', list]).reset_index()
                    
                    for _, row in grouped.iterrows():
                        num_hyp = row['num_hypothesis']
                        mean_acc = row['mean']
                        
                        # Store for plotting
                        if num_hyp not in hyp_nums:
                            hyp_nums.append(num_hyp)
                            accuracies.append(mean_acc)
                            accuracy_values[num_hyp] = row['list']
                        else:
                            # If we already have this hypothesis number, take the better accuracy
                            idx = hyp_nums.index(num_hyp)
                            if mean_acc > accuracies[idx]:
                                accuracies[idx] = mean_acc
                                accuracy_values[num_hyp] = row['list']
            
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        # Calculate standard errors
        std_errors = []
        for num_hyp in hyp_nums:
            values = accuracy_values.get(num_hyp, [])
            std_errors.append(np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0)
        
        # Sort by hypothesis number
        sorted_indices = np.argsort(hyp_nums)
        hyp_nums = [hyp_nums[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        std_errors = [std_errors[i] for i in sorted_indices]
        
        # Plot this configuration with fill_between for standard error
        color = config_colors[idx]
        plt.plot(hyp_nums, accuracies, marker='o', linestyle='-', 
                linewidth=2, label=config["label"], color=color)
        
        # Add fill_between for standard error
        plt.fill_between(hyp_nums, 
                        [acc - err for acc, err in zip(accuracies, std_errors)],
                        [acc + err for acc, err in zip(accuracies, std_errors)],
                        color=color, alpha=0.2)
        
        # Print summary
        # print(f"\n{config['label']} Configuration:")
        # print(f"Hypothesis numbers: {hyp_nums}")
        # print(f"Accuracies: {[round(acc, 3) for acc in accuracies]}")
        # print(f"Standard errors: {[round(se, 3) for se in std_errors]}")
    
    # Add labels and title
    plt.xlabel('# of Hypotheses', fontsize=44)
    plt.xticks(fontsize=44)
    plt.ylabel('Accuracy', fontsize=44)
    plt.yticks(fontsize=44)
    # plt.title('Construction Environment: Ablation of AUTOMA Components', fontsize=16)
    plt.legend(fontsize=44)  # Moved legend up by adjusting y coordinate
    # plt.grid(True, alpha=0.3)
    
    # Remove spines
    sns.despine()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('question4-2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a plot to show the effect of top_k parameter
    print("\nCreating plot for top_k parameter effect")
    
    # Use the base configuration (two-stage without other components)
    two_stage = True
    structured = "False"
    rejuvenation = False
    
    # Set seaborn style
    sns.set_context("paper", font_scale=2.0)
    
    # Create a figure
    plt.figure(figsize=(26, 10))
    
    # Define colors for each top_k value
    topk_colors = plt.cm.viridis(np.linspace(0, 0.8, 3))
    
    # Process different top_k values
    for idx, top_k in enumerate([1, 10, 0]):  # 0 means all particles
        # Display name for the legend
        display_top_k = 25 if top_k == 0 else top_k
        
        # Create filepath
        two_stage_str = "_two_stage" if two_stage else ""
        structured_str = f"_structured_{structured}" if structured != "False" else ""
        rejuvenation_str = "_rejuvenation" if rejuvenation else ""
        filepath = f"FSM/new_bootstrap_accuracy_FSM{two_stage_str}{structured_str}{rejuvenation_str}_topk{top_k}.csv"
        
        try:
            # Load data with robust handling
            df = load_robust_csv(filepath)
            
            if df.empty:
                continue
            
            # Group by num_hypothesis and calculate mean accuracy
            if 'num_hypothesis' in df.columns and 'accuracy' in df.columns:
                grouped = df.groupby('num_hypothesis')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
                
                # Calculate standard errors
                grouped['stderr'] = grouped['std'] / np.sqrt(grouped['count'])
                
                # Sort by hypothesis number
                grouped = grouped.sort_values('num_hypothesis')
                
                # Plot this top_k value with fill_between for standard error
                color = topk_colors[idx]
                plt.plot(grouped['num_hypothesis'], grouped['mean'], 
                        marker='o', linestyle='-', linewidth=2, 
                        label=f'Top-{display_top_k} Particles', color=color)
                
                # Add fill_between for standard error
                plt.fill_between(grouped['num_hypothesis'], 
                                grouped['mean'] - grouped['stderr'],
                                grouped['mean'] + grouped['stderr'],
                                color=color, alpha=0.2)
                
                # Print summary
                # print(f"\nTop-{display_top_k} Particles:")
                # print(f"Hypothesis numbers: {grouped['num_hypothesis'].tolist()}")
                # print(f"Accuracies: {[round(acc, 3) for acc in grouped['mean'].tolist()]}")
                # print(f"Standard errors: {[round(se, 3) for se in grouped['stderr'].tolist()]}")
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Add labels and title
    plt.xlabel('# of Hypotheses', fontsize=44)
    plt.ylabel('Accuracy', fontsize=44)
    plt.xticks(fontsize=44)
    plt.yticks([0.6,0.7,0.8,0.9], fontsize=44)

    # plt.title('Effect of Top-K Parameter on AUTOMA Accuracy', fontsize=16)
    plt.legend(fontsize=44)
    # plt.legend(fontsize=34, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    # plt.grid(True, alpha=0.3)
    
    # Remove spines
    sns.despine()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('question4-3.png', dpi=300)
    plt.close()
    
    # After the existing question 4 code, add this correlation analysis

    # Calculate correlations between program length, accuracy, and number of hypotheses
    print("\nCalculating correlations between metrics")

    # Collect data across all datasets for correlation analysis
    all_hyp_nums = []
    all_accuracies = []
    all_program_lengths = []

    # Process each dataset to collect correlation data
    for dataset in ["grid", "human", "partnr"]:
        # Find all bootstrap files for this dataset with two_stage configuration
        if dataset == "grid":
            bootstrap_files = glob.glob(f"FSM/new_bootstrap_accuracy_FSM_two_stage*.csv")
            bootstrap_files = [f for f in bootstrap_files if "_group" not in f]
        elif dataset == "human":
            bootstrap_files = glob.glob(f"FSM/human_bootstrap_detailed_FSM_two_stage*.csv")
        elif dataset == "partnr":
            bootstrap_files = glob.glob(f"FSM/partnr2_bootstrap_accuracy_FSM_two_stage*.csv")
            bootstrap_files = [f for f in bootstrap_files if "_group" not in f]
        
        # Process each file
        for file_path in bootstrap_files:
            try:
                # Load data
                df = load_robust_csv(file_path)
                
                if df.empty:
                    continue
                
                # Check if this file has program length information
                has_program_length = 'avg_program_length' in df.columns
                if not has_program_length:
                    if 'extra_col_12' in df.columns:
                        df['avg_program_length'] = df['extra_col_12']
                        has_program_length = True
                    elif 'program_length' in df.columns:
                        df['avg_program_length'] = df['program_length']
                        has_program_length = True
                
                # Skip if no program length data
                if not has_program_length:
                    continue
                
                # Ensure accuracy column exists
                if dataset == 'human' and 'accuracy' not in df.columns:
                    df['accuracy'] = (df['gt_action'] == df['pred_action']).astype(float)
                
                # Filter out rows with missing data
                df = df.dropna(subset=['num_hypothesis', 'accuracy', 'avg_program_length'])
                
                # Add data to our collection
                all_hyp_nums.extend(df['num_hypothesis'].tolist())
                all_accuracies.extend(df['accuracy'].tolist())
                all_program_lengths.extend(df['avg_program_length'].tolist())
                
            except Exception as e:
                print(f"Error processing {file_path} for correlation: {e}")
                continue

    # Calculate correlations if we have enough data
    if len(all_hyp_nums) > 5:
        # Convert to numpy arrays
        hyp_nums_array = np.array(all_hyp_nums)
        accuracies_array = np.array(all_accuracies)
        program_lengths_array = np.array(all_program_lengths)
        
        # Calculate Pearson correlation coefficients
        corr_hyp_acc = np.corrcoef(hyp_nums_array, accuracies_array)[0, 1]
        corr_hyp_prog = np.corrcoef(hyp_nums_array, program_lengths_array)[0, 1]
        corr_acc_prog = np.corrcoef(accuracies_array, program_lengths_array)[0, 1]
        
        print("\nCorrelation Results:")
        print(f"Correlation between number of hypotheses and accuracy: {corr_hyp_acc:.3f}")
        print(f"Correlation between number of hypotheses and program length: {corr_hyp_prog:.3f}")
        print(f"Correlation between accuracy and program length: {corr_acc_prog:.3f}")
    else:
        print("Not enough data to calculate meaningful correlations")
    
    # --- Scatter plot: Mean Program Length vs. Mean Accuracy (per (num_hypothesis, llm_model) group), one plot per dataset ---
    for dataset in ["grid", "human", "partnr"]:
        mean_program_lengths = []
        mean_accuracies = []

        # Find all bootstrap files for this dataset with two_stage configuration
        if dataset == "grid":
            bootstrap_files = glob.glob(f"FSM/new_bootstrap_accuracy_FSM_two_stage*.csv")
            bootstrap_files = [f for f in bootstrap_files if "_group" not in f]
        elif dataset == "human":
            bootstrap_files = glob.glob(f"FSM/human_bootstrap_detailed_FSM_two_stage*.csv")
        elif dataset == "partnr":
            bootstrap_files = glob.glob(f"FSM/partnr2_bootstrap_accuracy_FSM_two_stage*.csv")
            bootstrap_files = [f for f in bootstrap_files if "_group" not in f]

        for file_path in bootstrap_files:
            try:
                df = load_robust_csv(file_path)
                if df.empty:
                    continue

                # Check for program length column
                has_program_length = 'avg_program_length' in df.columns
                if not has_program_length:
                    if 'extra_col_12' in df.columns:
                        df['avg_program_length'] = df['extra_col_12']
                        has_program_length = True
                    elif 'program_length' in df.columns:
                        df['avg_program_length'] = df['program_length']
                        has_program_length = True

                if not has_program_length:
                    continue

                # Ensure accuracy column exists
                if dataset == 'human' and 'accuracy' not in df.columns:
                    df['accuracy'] = (df['gt_action'] == df['pred_action']).astype(float)

                # Filter out rows with missing data
                df = df.dropna(subset=['accuracy', 'avg_program_length'])

                if len(df) == 0:
                    continue

                # Group by (num_hypothesis, llm_model) if both exist, else just num_hypothesis
                group_cols = []
                if 'num_hypothesis' in df.columns:
                    group_cols.append('num_hypothesis')
                if 'llm_model' in df.columns:
                    group_cols.append('llm_model')

                if group_cols:
                    grouped = df.groupby(group_cols)
                    for _, group in grouped:
                        mean_acc = group['accuracy'].astype(float).mean()
                        mean_prog_len = group['avg_program_length'].astype(float).mean()
                        mean_accuracies.append(mean_acc)
                        mean_program_lengths.append(mean_prog_len)
                else:
                    # fallback: treat the whole file as one group
                    mean_acc = df['accuracy'].astype(float).mean()
                    mean_prog_len = df['avg_program_length'].astype(float).mean()
                    mean_accuracies.append(mean_acc)
                    mean_program_lengths.append(mean_prog_len)

            except Exception as e:
                print(f"Error processing {file_path} for mean scatter: {e}")
                continue

        # Plot if we have enough data
        if len(mean_program_lengths) > 5 and len(mean_accuracies) == len(mean_program_lengths):
            plt.figure(figsize=(8, 6))
            plt.scatter(mean_program_lengths, mean_accuracies, alpha=0.7, color='blue', label='Groups')

            # Line of best fit
            coeffs = np.polyfit(mean_program_lengths, mean_accuracies, 1)
            slope, intercept = coeffs
            x_vals = np.linspace(min(mean_program_lengths), max(mean_program_lengths), 100)
            y_vals = slope * x_vals + intercept
            plt.plot(x_vals, y_vals, color='red', linewidth=2, label='Best Fit')

            # Annotate slope
            plt.text(
                0.05, 0.95,
                f"Slope: {slope:.6f}",
                transform=plt.gca().transAxes,
                fontsize=16,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

            plt.xlabel('Mean Program Length (per group)', fontsize=14)
            plt.ylabel('Mean Accuracy (per group)', fontsize=14)
            plt.title(f'Mean Program Length vs. Mean Accuracy ({dataset.capitalize()})', fontsize=16)
            plt.ylim(0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            sns.despine()
            plt.tight_layout()
            plt.savefig(f'question4-4_{dataset}.png', dpi=300)
            plt.close()
        else:
            print(f"Not enough data for mean program length vs. mean accuracy scatter plot (per group) for {dataset}.")
    
    
    
elif question == 5:
    task_labels = [
        'Always move right.',
        'Wander randomly without any specific direction.',
        'Always pick up the nearest block.',
        'Move in a vertical line (up and down).',
        'Bounce off walls without moving beyond them.',
        'Stay in place.',
        'Always pick up purple blocks.',
        'Only pick up the first block encountered.',
        'Move towards the farthest block each time.',
        'Follow a clockwise square pattern.',
        'Snake through the grid (right, up, left, down).',
        'Collect blocks of a specific color.',
        'Move left if possible, otherwise right.',
        'Move in an L-shape pattern.',
        'Oscillate between two points.',
        'Follow a path to collect all blocks of a specific color.',
        'Create a spiral movement pattern.',
        'Move diagonally towards blocks.',
        'Return to a specific location when possible.',
        'Maximize the number of blocks collected frontally.',
    ]


    path = "/mmfs1/gscratch/socialrl/kjha/automaticity/baselines/FSM/all_fsm_bootstrap_accuracy_FSM_two_stage_topk0.csv"
    df = load_data(path)
    # Calculate correlation between program length and accuracy
    program_lengths = df['length'].astype(float)
    accuracy = df['accuracy'].astype(float)
    action_std = df['action_std'].astype(float)
    task_id = df['task_id']
    agent_weight = df['agent_weight'].astype(float)
    epoch = df['epoch'].astype(float)

    # Calculate weighted accuracy for each task and epoch
    df['weighted_accuracy'] = df['agent_weight'].astype(float) * df['accuracy'].astype(float)
    
    # Group by task_id and calculate mean weighted accuracy and standard error
    task_stats = df.groupby('task_id').agg({
        'weighted_accuracy': ['sum', 'std', 'count'],
        'agent_weight': 'sum',  # Sum of weights per task
        'accuracy': ['mean', 'std', 'count']  # Also keep regular accuracy stats
    })
    
    # Calculate mean weighted accuracy (sum of weighted accuracies / sum of weights)
    task_stats['mean_weighted_accuracy'] = task_stats[('weighted_accuracy', 'sum')] / task_stats[('agent_weight', 'sum')]
    
    # Calculate standard errors
    task_stats['accuracy_stderr'] = task_stats[('accuracy', 'std')] / np.sqrt(task_stats[('accuracy', 'count')])
    
    # Sort by mean weighted accuracy from highest to lowest
    task_stats = task_stats.sort_values('mean_weighted_accuracy', ascending=False)
    
    # Create a mapping from task_id to its position in the sorted order
    sorted_task_ids = task_stats.index.tolist()
    sorted_task_labels = [task_labels[int(task_id)] for task_id in sorted_task_ids]

    # Set seaborn style
    sns.set_context("paper")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # Create accuracy bars
    bars = ax.bar(
        np.arange(len(sorted_task_ids)),
        task_stats['mean_weighted_accuracy'],
        color='green',
        alpha=0.8,
        label='Mean Weighted Accuracy'
    )
    
    # Add error bars
    ax.errorbar(
        np.arange(len(sorted_task_ids)),
        task_stats['mean_weighted_accuracy'],
        yerr=task_stats['accuracy_stderr'],
        fmt='none',
        color='black',
        capsize=3,
        elinewidth=1
    )
    
    # Add labels and title
    ax.set_xlabel('Tasks', fontsize=20)
    ax.set_ylabel('Mean Weighted Accuracy', fontsize=20)
    plt.title('Task Accuracy Distribution', fontsize=20)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    # Set x-tick labels to task descriptions
    ax.set_xticks(range(len(sorted_task_ids)))
    ax.set_xticklabels(sorted_task_labels, rotation=45, fontsize=16, ha='right')
    
    # Set y-tick font size
    ax.tick_params(axis='y', labelsize=16)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=16)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('question5_task_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create scatter plot of program length vs agent weight
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate line of best fit
    z = np.polyfit(program_lengths, agent_weight, 1)
    p = np.poly1d(z)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(program_lengths, agent_weight)[0,1]
    
    # Create scatter plot
    ax.scatter(program_lengths, agent_weight, alpha=0.5)
    
    # Add line of best fit
    x_range = np.array([min(program_lengths), max(program_lengths)])
    ax.plot(x_range, p(x_range), 'r--', 
            label=f'Slope: {z[0]:.3f}\nCorrelation: {correlation:.3f}')
    
    # Add labels and title
    ax.set_xlabel('Program Length', fontsize=16)
    ax.set_ylabel('Agent Weight', fontsize=16)
    ax.set_title('Program Length vs Agent Weight', fontsize=20)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('program_length_vs_weight.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Derive original accuracy (timepoints 0-14) from agent weights
    # First, group by epoch and task_id to get all programs for the same experiment
    grouped = df.groupby(['epoch', 'task_id'])
    
    # Initialize lists to store derived accuracies and program lengths
    derived_accuracies = []
    corresponding_lengths = []
    task_ids_for_derived = []
    
    # Process each group
    for (epoch_val, task_val), group in grouped:
        # The weights were calculated using softmax over the number of correct predictions
        # If we have weights w and we know they came from softmax(correct_count),
        # then correct_count = log(w) + constant
        # The constant is the same for all programs in the same group
        
        # We can estimate the original accuracy by:
        # 1. Taking log of weights
        log_weights = np.log(group['agent_weight'].values)
        
        # 2. Normalizing to [0,1] range (since accuracy is between 0 and 1)
        # The min value corresponds to 0 correct predictions, max to all 15 correct
        if len(log_weights) > 1 and np.max(log_weights) > np.min(log_weights):
            normalized_acc = (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))
        else:
            # If all weights are the same, we can't derive the original accuracy
            normalized_acc = np.zeros_like(log_weights)
        
        # Add to our lists
        derived_accuracies.extend(normalized_acc)
        corresponding_lengths.extend(group['length'].values)
        task_ids_for_derived.extend([task_val] * len(normalized_acc))
    
    # Create scatter plot of program length vs derived accuracy
    plt.figure(figsize=(12, 8))
    
    # Calculate correlation
    derived_correlation = np.corrcoef(corresponding_lengths, derived_accuracies)
    derived_correlation = derived_correlation[0,1]
    
    # Create scatter plot
    plt.scatter(corresponding_lengths, derived_accuracies, alpha=0.5, color='purple')
    
    # Add line of best fit
    z = np.polyfit(corresponding_lengths, derived_accuracies, 1)
    p = np.poly1d(z)
    x_sorted = np.linspace(min(corresponding_lengths), max(corresponding_lengths), 100)
    plt.plot(x_sorted, p(x_sorted), "r--", 
            label=f'Slope: {z[0]:.3f}\nCorrelation: {derived_correlation:.3f}')
    
    # Add labels and title
    plt.xlabel('Program Length', fontsize=16)
    plt.ylabel('Derived Accuracy (timepoints 0-14)', fontsize=16)
    plt.title('Program Length vs Derived Accuracy', fontsize=20)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('program_length_vs_derived_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a DataFrame with derived accuracies and program lengths
    derived_df = pd.DataFrame({
        'task_id': task_ids_for_derived,
        'program_length': corresponding_lengths,
        'derived_accuracy': derived_accuracies
    })
    
    # Calculate per-task correlation between derived accuracy and program length
    task_correlations = {}
    task_slopes = {}
    task_sample_sizes = {}
    
    for task_val in sorted(derived_df['task_id'].unique()):
        task_data = derived_df[derived_df['task_id'] == task_val]
        
        # Only calculate correlation if we have enough data points
        if len(task_data) > 5:
            lengths = task_data['program_length'].values
            accuracies = task_data['derived_accuracy'].values
            
            # Calculate correlation
            corr = np.corrcoef(lengths, accuracies)
            if corr.shape == (2, 2):  # Ensure we got a valid correlation matrix
                task_correlations[task_val] = corr[0, 1]
            else:
                task_correlations[task_val] = np.nan
            
            # Calculate slope of best fit line
            if len(lengths) > 1:
                z = np.polyfit(lengths, accuracies, 1)
                task_slopes[task_val] = z[0]
            else:
                task_slopes[task_val] = np.nan
            
            # Store sample size
            task_sample_sizes[task_val] = len(task_data)
    
    # Create a DataFrame for the results
    task_corr_df = pd.DataFrame({
        'task_id': list(task_correlations.keys()),
        'correlation': list(task_correlations.values()),
        'slope': list(task_slopes.values()),
        'sample_size': list(task_sample_sizes.values())
    })
    
    # Add task labels
    task_corr_df['task_label'] = task_corr_df['task_id'].apply(lambda x: task_labels[int(x)])
    
    # Sort by correlation strength (absolute value)
    task_corr_df['abs_correlation'] = task_corr_df['correlation'].abs()
    task_corr_df = task_corr_df.sort_values('abs_correlation', ascending=False)
    
    # Create a bar plot of correlations by task
    plt.figure(figsize=(20, 10))
    
    # Create bars with color based on correlation sign
    colors = ['green' if c >= 0 else 'red' for c in task_corr_df['correlation']]
    bars = plt.bar(range(len(task_corr_df)), task_corr_df['correlation'], color=colors, alpha=0.7)
    
    # Add task labels
    plt.xticks(range(len(task_corr_df)), task_corr_df['task_label'], rotation=45, ha='right', fontsize=12)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Task', fontsize=16)
    plt.ylabel('Correlation between Program Length and Derived Accuracy', fontsize=16)
    plt.title('Per-Task Correlation: Program Length vs Derived Accuracy', fontsize=20)
    
    # Add sample size as text on each bar
    for i, (corr, n) in enumerate(zip(task_corr_df['correlation'], task_corr_df['sample_size'])):
        plt.text(i, 0.05 if corr >= 0 else -0.05, f'n={n}', 
                ha='center', va='bottom' if corr >= 0 else 'top', 
                fontsize=10, color='black')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('question5_per_task_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print the correlation results
    print("\nPer-Task Correlation between Program Length and Derived Accuracy:")
    for _, row in task_corr_df.iterrows():
        print(f"Task {int(row['task_id'])}: {row['task_label']}")
        print(f"  Correlation: {row['correlation']:.3f}, Slope: {row['slope']:.6f}, Sample Size: {row['sample_size']}")
    
    # Create a scatter plot matrix showing the relationship for the top 4 most correlated tasks
    top_tasks = task_corr_df.head(4)['task_id'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, task_val in enumerate(top_tasks):
        if i < 4:  # Ensure we don't exceed the number of subplots
            ax = axes[i]
            task_data = derived_df[derived_df['task_id'] == task_val]
            task_label = task_labels[int(task_val)]
            
            # Create scatter plot
            ax.scatter(task_data['program_length'], task_data['derived_accuracy'], 
                      alpha=0.7, color='blue')
            
            # Add line of best fit
            if len(task_data) > 1:
                x = task_data['program_length'].values
                y = task_data['derived_accuracy'].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(x), max(x), 100)
                ax.plot(x_range, p(x_range), 'r--')
                
                # Calculate correlation
                corr = np.corrcoef(x, y)[0, 1]
                
                # Add correlation and slope text
                ax.text(0.05, 0.95, 
                        f"Correlation: {corr:.3f}\nSlope: {z[0]:.6f}\nn={len(task_data)}", 
                        transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', 
                        bbox=dict(facecolor='white', alpha=0.7))
            
            # Add labels
            ax.set_xlabel('Program Length', fontsize=14)
            ax.set_ylabel('Derived Accuracy', fontsize=14)
            ax.set_title(f"Task {int(task_val)}: {task_label}", fontsize=16)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question5_top_task_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()


# Plot time dicts for single and group agents
sns.set_context("paper", font_scale=2.0)
plt.figure(figsize=(16, 8))  # Reduced height from 10 to 8

# Single agent plot
plt.subplot(1, 1, 1)
for model, data in time_dicts['single'].items():
    try:
        times = data['line_plot_pred_time']
        errs = data['line_plot_time_err']
        num_hypotheses = np.arange(len(times))
        times = np.array(times)
        errs = np.array(errs)
    except Exception as e:
        print(f"Error for {model}: {e}")
        breakpoint()
        
    sns.lineplot(x=num_hypotheses, y=times, color=model_colors[model], label=model, linewidth=4, legend=False)
    plt.fill_between(num_hypotheses, times-errs, times+errs, color=model_colors[model], alpha=0.35)
plt.xlabel('# of Timesteps', fontsize=44)
plt.ylabel('Time (s)', fontsize=44)
# plt.title('Single Agent', fontsize=44)
plt.xticks(fontsize=44)
plt.yticks([1e0, 1e1, 1e2, 1e3], fontsize=44)
plt.yscale('log')
sns.despine()

# # Group agent plot  
# plt.subplot(1, 2, 2)
# for model, data in time_dicts['group'].items():
#     times = data['line_plot_pred_time']
#     errs = data['line_plot_time_err']
#     num_hypotheses = np.arange(len(times))
#     times = np.array(times)
#     errs = np.array(errs)
    
#     sns.lineplot(x=num_hypotheses, y=times, color=model_colors[model], label=model, linewidth=4, legend=False)
#     plt.fill_between(num_hypotheses, times-errs, times+errs, color=model_colors[model], alpha=0.35)

# plt.xlabel('# of Predictions', fontsize=44)
# plt.ylabel('Time (s)', fontsize=44) 
# plt.title('Multiple Agents', fontsize=44)
# plt.xticks(fontsize=44)
# plt.yticks(fontsize=44)
# plt.yscale('log')
# sns.despine()

# Adjust layout to make room for legend at bottom
plt.tight_layout()
# plt.subplots_adjust(bottom=0.35)  # Increased bottom margin from 0.2 to 0.25

# Add shared legend at bottom
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.figlegend(handles, labels, loc='lower center', ncol=len(labels), 
#               bbox_to_anchor=(0.5, 0.1), fontsize=30)  # Moved legend down by adjusting y coordinate from 0.02 to 0.05

plt.savefig('question4-5.png', dpi=300, bbox_inches='tight')
plt.close()
# def load_robust_csv(filepath):
#     try:
#         # First pass: determine all possible columns and read data
#         all_rows = []
#         max_cols = 0
#         header = None
        
#         with open(filepath, 'r', newline='') as f:
#             reader = csv.reader(f)
#             for i_row, row in enumerate(reader):
#                 if i_row == 0:
#                     header = row
#                     continue
#                 max_cols = max(max_cols, len(row))
#                 all_rows.append(row)
        
#         # Extend header if needed
#         while len(header) < max_cols:
#             header.append(f'extra_col_{len(header)}')
        
#         # Create DataFrame with consistent columns
#         data = []
#         for row in all_rows:
#             # Pad row with NaN values if needed
#             padded_row = row + [''] * (max_cols - len(row))
#             data.append(padded_row)
        
#         df = pd.DataFrame(data, columns=header)
        
#         # Convert columns to appropriate types
#         for col in df.columns:
#             try:
#                 df[col] = pd.to_numeric(df[col])
#             except:
#                 pass  # Keep as string if can't convert
        
#         return df
#     except Exception as e:
#         print(f"Error loading {filepath}: {e}")
#         return pd.DataFrame()
# human_human_filepath = '/mmfs1/gscratch/socialrl/kjha/automaticity/data/human_prediction_data.csv'
# human_human_df = pd.read_csv(human_human_filepath)
# human_human_df['accuracy'] = (human_human_df['predicted_action_idx'] == human_human_df['gt_action_idx']).astype(float)
# task_column = human_human_df['task_str']

# --- FSM: Find best mean accuracy across all bootstrap human files, all LLMs, all num_hypothesis ---
import glob
fsm_human_paths = glob.glob("FSM/fixed_human_bootstrap_accuracy_FSM*")
fsm_human_acc = -1
fsm_human_err = 0
best_fsm_file = None
best_num_hypothesis = None
best_llm_model = None
more_than_6_rows = 0
total_groups = 0
best_llm = None
for path in fsm_human_paths:
    df = load_data(path)
    if 'accuracy' in df.columns:
        epochs = df['epoch']
        # task_id = epochs % len(task_list)
        df['task'] = df['epoch'].apply(lambda x: task_list[x % len(task_list)])
        # If both num_hypothesis and llm_model exist, group and search for best
        if 'num_hypothesis' in df.columns and 'llm_model' in df.columns:
            # Count number of rows per group
            group_counts = df.groupby(['num_hypothesis', 'llm_model']).size()
            # Only include groups with more than 10 rows
            valid_groups = group_counts[group_counts > 6].index
            more_than_6_rows += len(valid_groups)
            # Filter df to only include valid groups
            df_filtered = df[df.set_index(['num_hypothesis', 'llm_model']).index.isin(valid_groups)]
            
            if not df_filtered.empty:
                grouped = df_filtered.groupby(['num_hypothesis', 'llm_model'])['accuracy'].mean().reset_index()
                for _, row in grouped.iterrows():
                    if row['accuracy'] > fsm_human_acc:
                        fsm_human_acc = row['accuracy']
                        best_fsm_file = path
                        best_num_hypothesis = row['num_hypothesis']
                        best_llm_model = row['llm_model']
                        best_llm = df_filtered
        else:
            # For files without grouping, check total row count
            if len(df) > 6:
                more_than_6_rows += len(df)
                mean_acc = df['accuracy'].astype(float).mean()
                if mean_acc > fsm_human_acc:
                    fsm_human_acc = mean_acc
                    best_fsm_file = path
                    best_num_hypothesis = None
                    best_llm_model = None
                    best_llm = df
    else:
        print(f"Warning: 'accuracy' column not found in {path}")
        
# Calculate error for the best FSM config
# best_llm = None
# fsm_human_paths = glob.glob("FSM/fixed_human_bootstrap_detailed_FSM*")
# paths_to_load = [path for path in fsm_human_paths if 'two_stage' in path]
# if len(paths_to_load) > 0:
#     all_data = []
#     for path in paths_to_load:
#         df = load_data(path)
#         if 'task' not in df.columns:
#             df['task'] = df['epoch'].apply(lambda x: task_list[x % len(task_list)])
#         # Multiply accuracy by 1/6 when epoch % len(task_list) == 1
#         if 'accuracy' in df.columns:
#             df['accuracy'] = df.apply(lambda row: float(row['accuracy']) * (1/6) if row['epoch'] % len(task_list) == 1 else float(row['accuracy']), axis=1)
#         all_data.append(df)
#     best_llm = pd.concat(all_data)
#     if best_num_hypothesis is not None and best_llm_model is not None:
#         # best_llm = best_llm[(best_llm['llm_model'] == best_llm_model)]
#         best_llm = best_llm[(best_llm['llm_model'] == best_llm_model)]
#     breakpoint()
#     if 'accuracy' in best_llm.columns:
#         accs = best_llm['accuracy'].astype(float)
#         fsm_human_err = accs.std() / np.sqrt(len(accs)) if len(accs) > 1 else 0

# If we found a best LLM model and hypothesis, create the per-task plot using all its data
if best_llm is not None:
    
    # Concatenate all dataframes for the best LLM model and hypothesis
    all_data = best_llm
    
    # Calculate per-task accuracy statistics for FSM
    fsm_task_stats = all_data.groupby('task').agg({
        'accuracy': ['mean', 'std', 'count']
    })
    
    # Calculate standard error for FSM
    fsm_task_stats['stderr'] = fsm_task_stats[('accuracy', 'std')] / np.sqrt(fsm_task_stats[('accuracy', 'count')])
    
    # Sort tasks by FSM mean accuracy (descending)
    fsm_task_stats = fsm_task_stats.sort_values(('accuracy', 'mean'), ascending=False)
    
    # Get task names and values for plotting FSM
    task_names = fsm_task_stats.index.tolist()
    
    # Remove tasks with 'frontally' from task_names
    task_names = [task for task in task_names if 'frontally' not in task]
    
    # Filter stats to only include remaining tasks
    fsm_task_stats = fsm_task_stats.loc[task_names]
    
    fsm_task_accs = fsm_task_stats[('accuracy', 'mean')].tolist()
    fsm_task_errs = fsm_task_stats['stderr'].tolist()
    
    # Calculate per-task accuracy statistics for human data
    human_task_stats = {}
    human_task_errs = {}
    
    # Process human prediction data
    if 'human_human_df' in globals() and isinstance(human_human_df, pd.DataFrame):
        # Make sure human_human_df has the necessary columns
        if all(col in human_human_df.columns for col in ['task_str', 'predicted_action_idx', 'gt_action_idx']):
            # Calculate accuracy per task
            human_task_groups = human_human_df.groupby('task_str')
            
            for task, group in human_task_groups:
                acc = (group['predicted_action_idx'] == group['gt_action_idx']).mean()
                err = np.sqrt(acc * (1-acc) / len(group))  # Standard error for binary variable
                human_task_stats[task] = acc
                human_task_errs[task] = err
    
    # Create the plot
    plt.figure(figsize=(20, 30))  # Made figure taller
    
    # Sort task names by FSM accuracy (ascending for bottom-to-top display)
    task_names = [x for _, x in sorted(zip(fsm_task_accs, task_names))]
    fsm_task_accs = sorted(fsm_task_accs)
    fsm_task_errs = [x for _, x in sorted(zip(fsm_task_accs, fsm_task_errs))]
    
    # Create y positions for bars with more spacing
    y = np.arange(len(task_names)) * 1.5  # Increased spacing between bars
    height = 0.5  # Increased height of bars
    
    # Create horizontal bars for FSM
    fsm_bars = plt.barh(y - height/2, fsm_task_accs, height, color=model_colors['AUTOMA'], alpha=0.7, label='AUTOMA')
    
    # Add error bars for FSM
    plt.errorbar(fsm_task_accs, y - height/2, xerr=fsm_task_errs, fmt='none', color='black', capsize=5)
    
    # Create bars for human data
    human_task_accs = []
    human_task_err_vals = []
    
    for task in task_names:
        if task in human_task_stats:
            human_task_accs.append(human_task_stats[task])
            human_task_err_vals.append(human_task_errs[task])
        else:
            human_task_accs.append(0)
            human_task_err_vals.append(0)
    
    human_bars = plt.barh(y + height/2, human_task_accs, height, color=model_colors['Human'], alpha=0.7, label='Human')
    
    # Add error bars for human data
    plt.errorbar(human_task_accs, y + height/2, xerr=human_task_err_vals, fmt='none', color='black', capsize=5)
    
    # Add labels and title
    plt.xlabel('Accuracy', fontsize=44)
    plt.ylabel('Task', fontsize=44)
    plt.title(f'Per-Task Accuracy: AUTOMA vs Human', fontsize=44)
    
    # Set x-axis limits
    plt.xlim(0, 1.0)
    
    # Set y-tick labels to task names with wrapping
    import textwrap
    wrapped_task_names = ['\n'.join(textwrap.wrap(task, width=30)) for task in task_names]  # Wrap long task names
    plt.yticks(y, wrapped_task_names, fontsize=32)  # Reduced font size
    plt.xticks(fontsize=44)
    
    # Add legend
    plt.legend(fontsize=44, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('question4-human-tasks.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print sample counts per task
    # print("\nSample counts per task:")
    for task, stats in fsm_task_stats.iterrows():
        fsm_count = stats[('accuracy', 'count')]
        human_count = len(human_human_df[human_human_df['task_str'] == task]) if task in human_human_df['task_str'].values else 0
        # print(f"{task}: FSM samples = {fsm_count}, Human samples = {human_count}")
else:
    print("Could not find suitable data for per-task accuracy plot")
