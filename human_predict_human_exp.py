import asyncio
import os
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from nicegui import app, ui

import nicewebrl
from environment_jax import AutomaticityEnv, State, state_to_image_jit
from nicewebrl.logging import get_logger

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================
# Set this flag to False to use videos generated from human gameplay.
USE_FSM_DATA = False
PREDICTION_STEPS = 5 # The number of future steps the user will predict

logger = get_logger(__name__)

# Environment variables for configuration
NAME = os.environ.get('NAME', 'prediction_exp_gameplay')
DEBUG = int(os.environ.get('DEBUG', 0))
DATA_DIR = os.environ.get('DATA_DIR', 'human_gameplay_pred_data')
HUMAN_DATA_ROOT = 'human_gameplay_videos'
TUTORIAL_DATA_ROOT = 'human_vids_curated'


task_list = [
    "Repeatedly go from the green-->blue-->purple blocks",
    "Patrol the border of the grid in a clockwise manner",
    "Patrol the border of the grid in a counter-clockwise manner",
    "Alternate between going left until you hit a wall, then going right until you hit a wall",
    "Pickup a blue block and place it next to another blue block",
    "Repeatedly move in an L-shape pattern",
    "Pickup the green blocks and move them to a corner of the grid",
    "Pickup the pink blocks and move them to a corner of the grid",
    "Snake up and down across the rows of the grid",
    "Alternate between going up until you hit a wall, then going down until you hit a wall",
]

action_to_name = ["NOOP", "Right", "Left", "Down", "Up", "Pick up/Drop block"]

# ==============================================================================
#                               DATA LOADING
# ==============================================================================
def load_gameplay_data(data_path: str):
    """Loads the pickled LIST of trajectory data."""
    logger.info(f"Attempting to load gameplay data from {data_path}...")
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        logger.info("Successfully loaded gameplay data from pickle file.")
        return data
    except Exception as e:
        logger.error(f"Could not load or parse gameplay data: {e}", exc_info=True)
        return None

def get_trajectory_data(all_trajectories, video_file_prefix):
    """Helper to get the data for a single trajectory using its filename."""
    try:
        parts = video_file_prefix.split('_')
        variant_idx = int(parts[1])
        task_idx = int(parts[2])
        trajectory = all_trajectories[variant_idx][task_idx]
        if trajectory is None:
            logger.error(f"Trajectory data is None for '{video_file_prefix}'")
            return None
        return trajectory
    except (IndexError, ValueError, KeyError) as e:
        logger.error(f"Could not find data for '{video_file_prefix}': {e}")
        return None

def get_ground_truth_actions(trajectory_data):
    """Extracts the ground truth actions that occur after the first step."""
    try:
        actions_array = trajectory_data['actions']
        start_idx = 1
        end_idx = start_idx + PREDICTION_STEPS
        return actions_array[start_idx:end_idx]
    except Exception as e:
        logger.error(f"Could not extract ground truth actions: {e}")
        return None

# --- Main data loading section ---
DATA_LOADED = False
try:
    if USE_FSM_DATA:
        FSM_DATA_FILE = os.path.join(TUTORIAL_DATA_ROOT, 'fsm_curated_gameplay_data.pkl')
        all_trajectories = load_gameplay_data(FSM_DATA_FILE)
    else:
        HUMAN_DATA_FILE = os.path.join(HUMAN_DATA_ROOT, 'human_curated_gameplay_data.pkl')
        all_trajectories = load_gameplay_data(HUMAN_DATA_FILE)

    if all_trajectories is not None:
        DATA_LOADED = True
        logger.info(f"Data loading successful for {len(all_trajectories[0])} tasks across {len(all_trajectories)} variants.")
    else:
        raise ValueError("Data loading returned None. Check logs for errors.")
except Exception as e:
    logger.error(f"FATAL: Could not load gameplay data: {e}")
    all_trajectories = None
    DATA_LOADED = False

# ==============================================================================
#                               ENVIRONMENT SETUP
# ==============================================================================
def setup_prediction_environment():
    """Set up the 7x7 environment for multi-step predictions"""
    GRID_SIZE = 7
    TILE_SIZE = 12
    return AutomaticityEnv(num_agents=1, size=GRID_SIZE, max_steps=30, num_blocks=10, num_walls=30), GRID_SIZE, TILE_SIZE

prediction_env, GRID_SIZE, TILE_SIZE = setup_prediction_environment()

def load_prediction_data(video_file_prefix):
    """
    Loads states and actions.
    REVISION: Ensures all loaded state data uses JAX arrays, not NumPy arrays.
    """
    if not DATA_LOADED:
        logger.warning(f"Data not loaded, using random states for {video_file_prefix}")
        _, state_t0 = prediction_env.reset(jax.random.PRNGKey(hash(video_file_prefix)))
        _, state_t1 = prediction_env.step(state_t0, jnp.array([0]))
        return state_t0, state_t1, state_t1, None

    trajectory_data = get_trajectory_data(all_trajectories, video_file_prefix)
    if trajectory_data is None:
        logger.warning(f"Could not find trajectory data for {video_file_prefix}. Using random fallback.")
        _, state_t0 = prediction_env.reset(jax.random.PRNGKey(hash(video_file_prefix)))
        _, state_t1 = prediction_env.step(state_t0, jnp.array([0]))
        return state_t0, state_t1, state_t1, None

    try:
        states_pytree = trajectory_data['states']
        
        # 1. Extract the state data for timesteps 0 and 1 as a dictionary.
        state_dict_t0 = jax.tree.map(lambda x: x[0], states_pytree)
        state_dict_t1 = jax.tree.map(lambda x: x[1], states_pytree)

        # 2. **CRITICAL FIX**: Convert all NumPy arrays to JAX arrays.
        # This ensures the data type matches what the JIT-compiled rendering
        # function expects, preventing the TracerArrayConversionError.
        state_dict_t0_jax = jax.tree.map(lambda leaf: jnp.asarray(leaf), state_dict_t0)
        state_dict_t1_jax = jax.tree.map(lambda leaf: jnp.asarray(leaf), state_dict_t1)
        
        # 3. Convert the dictionaries of JAX arrays into official State objects.
        state_t0 = State(**state_dict_t0_jax)
        state_t1 = State(**state_dict_t1_jax)
        
        ground_truth = get_ground_truth_actions(trajectory_data)
        initial_prediction_state = state_t1
        
        return state_t0, state_t1, initial_prediction_state, ground_truth
        
    except Exception as e:
        logger.error(f"Failed to load prediction start frames for {video_file_prefix}: {e}", exc_info=True)
        _, state_t0 = prediction_env.reset(jax.random.PRNGKey(hash(video_file_prefix)))
        _, state_t1 = prediction_env.step(state_t0, jnp.array([0]))
        return state_t0, state_t1, state_t1, None

# ==============================================================================
#                               DATA STORAGE FUNCTIONS
# ==============================================================================
def store_stage_response(stage_idx, stage_data):
    if "all_stage_responses" not in app.storage.user:
        app.storage.user["all_stage_responses"] = []
    stage_response = {'stage_idx': stage_idx, 'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"), **stage_data}
    app.storage.user["all_stage_responses"].append(stage_response)
    for key, value in stage_data.items():
        app.storage.user[key] = value
    logger.info(f"Stored stage {stage_idx} response. Total stages stored: {len(app.storage.user['all_stage_responses'])}")

def get_user_save_file_fn():
    if "user_id" not in app.storage.user:
        app.storage.user["user_id"] = str(uuid.uuid4())[:8]
    if "session_timestamp" not in app.storage.user:
        app.storage.user["session_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'predict_user={app.storage.user["user_id"]}_session={app.storage.user["session_timestamp"]}_name={NAME}_debug={DEBUG}.json'
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)

# ==============================================================================
#                               DEBUGGING FUNCTIONS
# ==============================================================================
def print_state_debug_info(state: State, step_message: str):
    if not DEBUG: return
    try:
        print("\n" + "="*60); print(f"DEBUGGING STATE: {step_message}"); print("="*60)
        if hasattr(state, 'agent_locations'):
            agent_loc = state.agent_locations[0]
            print(f"  -> Agent Location: (col={int(agent_loc[0])}, row={int(agent_loc[1])})")
        else:
            print("  -> Agent Location: Not found in state object."); agent_loc = None
        if hasattr(state, 'block_locations') and hasattr(state, 'block_colors'):
            print("  -> Block Details:")
            for i in range(len(state.block_locations)):
                loc = state.block_locations[i]
                if loc[0] == -1: continue
                color = state.block_colors[i]; inventory_str = ""
                if agent_loc is not None and hasattr(state, 'agent_inventory') and state.agent_inventory[0] != -1:
                    if int(agent_loc[0]) == int(loc[0]) and int(agent_loc[1]) == int(loc[1]):
                        inventory_str = " (Carried by Agent)"
                print(f"     - Block {i}: Location=(col={int(loc[0])}, row={int(loc[1])}), Color=({int(color[0])}, {int(color[1])}, {int(color[2])}){inventory_str}")
        else:
            print("  -> Block Details: Not found in state object.")
        print("="*60 + "\n")
    except Exception as e:
        print(f"[DEBUG ERROR] Could not print state details: {e}")
        print(f"[DEBUG INFO] Available attributes in state object: {dir(state)}")

# ==============================================================================
#                             MULTI-STEP PREDICTION INTERFACE
# ==============================================================================
async def multistep_prediction_interface(container, initial_state, num_steps=5):
    nicewebrl.clear_element(container)
    print_state_debug_info(initial_state, "Initial Prediction State (Step 0)")
    current_step = {"value": 0}; predicted_actions = {"actions": []}; state_history = {"states": [initial_state]}
    current_state = {"state": initial_state}; processing_action = {"value": False}

    with container.style("align-items: center;"):
        ui.markdown(f"## Predict the Agent's Next {num_steps} Moves")
        ui.markdown(f"Click buttons to predict the next **{num_steps} moves**.")
        progress_container = ui.column().classes('items-center'); env_container = ui.column().style("align-items: center;")
        button_container = ui.column().style("align-items: center;"); summary_container = ui.column().style("align-items: center; min-height: 80px;")
        completion_event = asyncio.Event()

        def get_agent_positions_from_states():
            positions = []
            for state in state_history["states"]:
                try: positions.append((int(state.agent_locations[0][0]), int(state.agent_locations[0][1])))
                except: positions.append((0, 0))
            return positions

        def render_state_with_trajectory(state):
            try:
                obs = prediction_env.get_observation(state)[0]
                base_image = state_to_image_jit(obs, GRID_SIZE, TILE_SIZE)

                if len(state_history["states"]) <= 1:
                    final_image = base_image
                else:
                    trajectory_positions = get_agent_positions_from_states()
                    enhanced_image = jnp.array(base_image, dtype=jnp.float32)
                    for step_idx, pos in enumerate(trajectory_positions[:-1]):
                        x, y = pos
                        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                            pixel_y_start, pixel_x_start = y * TILE_SIZE, x * TILE_SIZE
                            for dy in range(TILE_SIZE):
                                for dx in range(TILE_SIZE):
                                    py, px = pixel_y_start + dy, pixel_x_start + dx
                                    enhanced_image = enhanced_image.at[py, px, 0].set(jnp.clip(enhanced_image[py, px, 0] * 0.7 + 60, 0, 255))
                                    enhanced_image = enhanced_image.at[py, px, 2].set(jnp.clip(enhanced_image[py, px, 2] * 0.7 + 120, 0, 255))
                            if step_idx > 0:
                                center_y, center_x = pixel_y_start + TILE_SIZE // 2, pixel_x_start + TILE_SIZE // 2
                                diamond_size = max(2, TILE_SIZE // 6)
                                for dy in range(-diamond_size, diamond_size + 1):
                                    for dx in range(-diamond_size, diamond_size + 1):
                                        py, px = center_y + dy, center_x + dx
                                        if (0 <= py < enhanced_image.shape[0] and 0 <= px < enhanced_image.shape[1]) and (abs(dx) + abs(dy) <= diamond_size):
                                            color = [255, 255, 255] if abs(dx) + abs(dy) <= diamond_size - 1 else [180, 100, 220]
                                            enhanced_image = enhanced_image.at[py, px].set(jnp.array(color))
                    final_image = enhanced_image.astype(jnp.uint8)
                scaled_image = jnp.repeat(jnp.repeat(final_image, repeats=3, axis=0), repeats=3, axis=1)
                return nicewebrl.base64_npimage(scaled_image.astype(jnp.uint8))
            except Exception as e:
                logger.error(f"Error rendering trajectory: {e}"); return None

        def update_display():
            nicewebrl.clear_element(progress_container); nicewebrl.clear_element(env_container); nicewebrl.clear_element(summary_container)
            with progress_container: ui.markdown(f"**Step {current_step['value']} of {num_steps}**")
            with env_container:
                state_image_b64 = render_state_with_trajectory(current_state['state'])
                if state_image_b64:
                    ui.html(f'<img src="{state_image_b64}" style="max-width: 500px; height: auto; border: 3px solid #333; border-radius: 8px;">')
                    with ui.column().style('min-height: 45px; display: flex; align-items: center; justify-content: center;'):
                        ui.html("""<div style="margin: 10px 0; padding: 8px; background: #f0f0f0; border-radius: 6px; font-size: 13px; text-align: center;"><span style="color: #9575cd;">■</span> Previous path&nbsp;&nbsp; <span style="color: #b464dc;">◆</span> Step markers</div>""")
            with summary_container:
                if predicted_actions['actions']: ui.markdown(f"**Predicted moves:** {' → '.join([action_to_name[a] for a in predicted_actions['actions']])}")

        async def handle_action(action_idx):
            if processing_action['value'] or current_step['value'] >= num_steps: return
            processing_action['value'] = True
            try:
                predicted_actions['actions'].append(action_idx)
                _, new_state = prediction_env.step(current_state['state'], jnp.array([action_idx]))
                current_state['state'] = new_state; state_history['states'].append(new_state)
                print_state_debug_info(new_state, f"After Action '{action_to_name[action_idx]}' (New State for Step {current_step['value'] + 1})")
                current_step['value'] += 1; update_display()
                if current_step['value'] >= num_steps: finish_prediction()
            finally: await asyncio.sleep(0.15); processing_action['value'] = False

        def undo_last_move():
            if processing_action['value'] or current_step['value'] <= 0: return
            processing_action['value'] = True
            try:
                predicted_actions['actions'].pop(); state_history['states'].pop(); current_step['value'] -= 1
                current_state['state'] = state_history['states'][-1]
                print_state_debug_info(current_state['state'], f"After UNDO (Reverted to State for Step {current_step['value']})")
                update_display()
                if current_step['value'] < num_steps: show_action_buttons()
            finally: asyncio.create_task(reset_processing_flag())

        async def reset_processing_flag(): await asyncio.sleep(0.15); processing_action['value'] = False

        def finish_prediction():
            nicewebrl.clear_element(button_container)
            with button_container:
                ui.markdown("### ✅ Prediction Complete!")
                ui.markdown(f"**Sequence:** {' → '.join([action_to_name[a] for a in predicted_actions['actions']])}")
                with ui.row().style("justify-content: center; gap: 15px; margin: 15px 0;"):
                    ui.button("↺ Undo", on_click=undo_last_move).props("color=warning outline")
                    ui.button("✓ Continue", on_click=lambda: completion_event.set()).props("color=primary size=lg")

        def show_action_buttons():
            nicewebrl.clear_element(button_container)
            with button_container:
                ui.markdown("**Choose an action:**")
                with ui.column().style("align-items: center; gap: 8px;"):
                    ui.button("↑ Up", on_click=lambda: asyncio.create_task(handle_action(4))).props("color=primary size=lg outline")
                    with ui.row().style("gap: 10px;"):
                        ui.button("← Left", on_click=lambda: asyncio.create_task(handle_action(2))).props("color=primary size=lg outline")
                        ui.button("↓ Down", on_click=lambda: asyncio.create_task(handle_action(3))).props("color=primary size=lg outline")
                        ui.button("→ Right", on_click=lambda: asyncio.create_task(handle_action(1))).props("color=primary size=lg outline")
                    with ui.row().style("gap: 10px; margin-top: 5px; align-items: center;"):
                        ui.button("⏹ Stay", on_click=lambda: asyncio.create_task(handle_action(0))).props("color=secondary outline")
                        ui.button("🔄 Pick up/Drop block", on_click=lambda: asyncio.create_task(handle_action(5))).props("color=accent outline")
                        ui.button("↺ Undo", on_click=undo_last_move).props("color=warning outline").bind_enabled_from(current_step, 'value', lambda v: v > 0)

        update_display(); show_action_buttons()
        await completion_event.wait()
        return predicted_actions['actions']

# ==============================================================================
#                               EXPERIMENT CREATION
# ==============================================================================
async def prediction_display_fn_with_state_loading(stage, container):
    ui.run_javascript("window.scrollTo(0, 0); document.body.scrollTop = 0; document.documentElement.scrollTop = 0;")
    nicewebrl.clear_element(container)
    demo_files = stage.metadata['demo_files']; predict_file = stage.metadata['predict_file']
    logger.info(f"Stage (Task Index {stage.metadata['task_idx']}): Displaying demos {demo_files}, predict {predict_file}")

    with container.style("align-items: center; width: 100%;"):
        ui.markdown(f"## {stage.name}")
        ui.markdown("### Part 1: Observe Agent Behavior")
        ui.markdown("Watch the following **three examples** of the agent's behavior. The agent is performing the same underlying task in each video, but the environment is different. **You can replay the video as many times as you want.**")
        with ui.row().classes('w-full no-wrap justify-center items-start').style("gap: 15px; margin-top: 20px;"):
            for i, file_prefix in enumerate(demo_files):
                with ui.column().style("align-items: center; max-width: 280px;"):
                    ui.markdown(f"**Example {i+1}**")
                    demo_video_path = f"/{HUMAN_DATA_ROOT}/{file_prefix}.mp4"
                    ui.video(demo_video_path, autoplay=False, muted=True, controls=True).style("width: 280px;")
        demos_watched = asyncio.Event()
        ui.button("I have watched the examples. Proceed to the prediction task.", on_click=lambda: (demos_watched.set(), ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)"))).props("color=primary size=lg").style("margin-top: 25px;")
        await demos_watched.wait()

        ui.markdown("---").style("width: 80%; margin: 30px auto;")
        ui.markdown("### Part 2: Predict the Agent's Next Moves")
        ui.markdown(f"Now, you will see the starting state and the next frame for a new trial. Based on what you learned, predict the agent's next **{PREDICTION_STEPS} moves**.")
        state_t0, state_t1, initial_prediction_state, ground_truth_actions = load_prediction_data(predict_file)

        with ui.row(wrap=False).classes('w-full justify-center items-center').style("gap: 20px; margin: 20px 0;"):
            with ui.column(align_items='center'):
                ui.markdown("**Start Frame (Time=0)**")
                obs_t0 = prediction_env.get_observation(state_t0)[0]
                img_t0 = state_to_image_jit(obs_t0, GRID_SIZE, TILE_SIZE)
                scaled_img_t0 = np.repeat(np.repeat(img_t0, 3, 0), 3, 1)
                ui.image(nicewebrl.base64_npimage(scaled_img_t0)).style("border: 2px solid #ccc; border-radius: 4px; width: 250px;")
            ui.icon('arrow_forward', size='xl', color='grey')
            with ui.column(align_items='center'):
                ui.markdown("**Next Frame (Time=1)**")
                obs_t1 = prediction_env.get_observation(state_t1)[0]
                img_t1 = state_to_image_jit(obs_t1, GRID_SIZE, TILE_SIZE)
                scaled_img_t1 = np.repeat(np.repeat(img_t1, 3, 0), 3, 1)
                ui.image(nicewebrl.base64_npimage(scaled_img_t1)).style("border: 2px solid #333; border-radius: 4px; width: 250px;")
        prediction_started = asyncio.Event()
        ui.button("Start Prediction", on_click=lambda: (prediction_started.set(), ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)"))).props("color=positive size=lg")
        await prediction_started.wait()
        prediction_container = ui.column().style("width: 100%;")
        predicted_actions = await multistep_prediction_interface(container=prediction_container, initial_state=initial_prediction_state, num_steps=PREDICTION_STEPS)

    nicewebrl.clear_element(container)
    ui.run_javascript("window.scrollTo(0, 0);")
    with container.style("align-items: center; width: 100%;"):
        ui.markdown("### Part 3: Rate the Agent's Behavior")
        ui.markdown("*Please answer the following questions about the agent's behavior based on all the videos you've seen for this task.*")
        ui.markdown("*Each question is rated on a 7-point scale.*")

        def gen_slider(question: str):
            ui.html(f'<div style="text-align: left; width: 100%; margin: 15px 0 10px 0; font-weight: normal;">{question}</div>')
            choices = {1: "Not at all", 2: "Slightly", 3: "Somewhat", 4: "Moderately", 5: "Quite a bit", 6: "Very much", 7: "Extremely"}
            rating = {"value": None}
            ui.toggle(choices, on_change=lambda e: rating.update({"value": e.value})).style("margin-bottom: 20px;")
            return rating

        ratings = {
            "predictable": gen_slider("How <b>predictable</b> was the agent's behavior?"),
            "robotic": gen_slider("How <b>scripted/robotic</b> did the agent's actions appear overall?"),
            "goal_directed": gen_slider("Did it seem that the agent was <b>planning ahead to pursue a goal</b>?"),
            "confidence": gen_slider(f"How <b>confident</b> are you in <b>predicting this agent's next {PREDICTION_STEPS} moves</b>?")
        }
        ui.markdown("### Behavior Description")
        ui.html("""<div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 20px 0;"><p><strong>Based on all the videos, how would you describe the behavior to another participant in this tasks such that they will be able to predict the agent's movement?</strong></p><p>Focus on the specific patterns or rules.</p></div>""")
        behavior_description = ui.textarea(label="Your description (1-2 sentences):").style("width: 80%;")
        success = False
        def submit():
            nonlocal success
            if any(r["value"] is None for r in ratings.values()) or not behavior_description.value.strip():
                ui.notify("Please answer all rating questions and provide a description before submitting.", type="warning")
                return
            gt_actions_list = ground_truth_actions.tolist() if ground_truth_actions is not None and hasattr(ground_truth_actions, 'tolist') else []
            stage_data = {
                "predicted_actions_sequence": predicted_actions, "ground_truth_actions": gt_actions_list,
                "demo_video_files": demo_files, "prediction_video_file": predict_file,
                "chosen_video_file": predict_file, "behavior_description": behavior_description.value,
                "task_idx": stage.metadata.get('task_idx'), "task_name": stage.metadata.get('task'),
                **{f"{name}_rating": r["value"] for name, r in ratings.items()}
            }
            store_stage_response(stage.metadata.get('task_idx', -1), stage_data)
            success = True
        submit_button = ui.button("Submit and Continue to Next Task", on_click=submit).props("color=primary size=lg")
        while not success: await submit_button.clicked()
    await stage.set_user_data(finished=True)

def create_tutorial_stage():
    async def tutorial_display_fn(stage, container):
        nicewebrl.clear_element(container); ui.run_javascript("window.scrollTo(0, 0);")
        AGENT_COLOR = "#D32F2F"; BLOCK_COLOR_1 = "#32D219"; BLOCK_COLOR_2 = "#1976D2"; WALL_COLOR = "#616161"
        with container.style("max-width: 1000px; margin: auto; padding: 20px;"):
            with ui.card().classes('w-full'):
                with ui.card_section(): ui.markdown("## Welcome to the Experiment!"); ui.markdown("In this experiment, you will observe an agent's behavior to understand its goals. First, let's learn the rules of its world and your task.")
                ui.separator()
                with ui.card_section():
                    ui.html('<style>.tutorial-card { min-height: 750px; }</style>')
                    with ui.row().classes('w-full q-gutter-md'):
                        with ui.column().classes('col-5'):
                            with ui.card().classes('w-full h-full tutorial-card'):
                                with ui.card_section():
                                    ui.markdown("#### 1. Basic Movement"); ui.markdown(f"The <span style='color: {AGENT_COLOR};'>**agent**</span> moves one square at a time using the action buttons.")
                                    with ui.row().classes('q-gutter-sm q-mt-xs'): ui.chip('Up', icon='arrow_upward'); ui.chip('Down', icon='arrow_downward'); ui.chip('Left', icon='arrow_back'); ui.chip('Right', icon='arrow_forward'); ui.chip('Stay', icon='pause')
                                with ui.card_section().classes('q-pt-none'): ui.video(f'/{TUTORIAL_DATA_ROOT}/tutorial/movement.mp4', autoplay=True, loop=True, muted=True).classes('w-full rounded-borders shadow-2')
                        with ui.column().classes('col-5'):
                            with ui.card().classes('w-full h-full tutorial-card'):
                                with ui.card_section():
                                    ui.markdown("#### 2. Walls are Obstacles"); ui.markdown(f"The <span style='color: {AGENT_COLOR};'>**agent**</span> cannot move through solid gray <span style='color: {WALL_COLOR};'>**walls**</span>.")
                                with ui.card_section().classes('q-pt-none'): ui.video(f'/{TUTORIAL_DATA_ROOT}/tutorial/walls.mp4', autoplay=True, loop=True, muted=True).classes('w-full rounded-borders shadow-2')
                    with ui.row().classes('w-full q-gutter-md q-mt-md'):
                        with ui.column().classes('col-5'):
                            with ui.card().classes('w-full h-full tutorial-card'):
                                with ui.card_section():
                                    ui.markdown("#### 3. Picking Up & Dropping Blocks"); ui.markdown(f"To **Pick Up** a <span style='color: {BLOCK_COLOR_1};'>**block**</span>, the <span style='color: {AGENT_COLOR};'>**agent**</span> must first move onto its square. Then, you must use the 'Pick up/Drop' action."); ui.markdown("To **Drop** a held <span style='color: {BLOCK_COLOR_1};'>**block**</span>, move to an empty square and use the same action again.")
                                    ui.chip('Pick up/Drop', icon='swap_horiz').props('color=accent text-color=white q-mt-sm'); ui.markdown(f"<br><b>Visual Indicator:</b> When the <span style='color: {AGENT_COLOR};'>**agent**</span> is carrying a <span style='color: {BLOCK_COLOR_1};'>**block**</span>, a small colored square appears in its center.")
                                with ui.card_section().classes('q-pt-none'): ui.video(f'/{TUTORIAL_DATA_ROOT}/tutorial/pickup_drop.mp4', autoplay=True, loop=True, muted=True).classes('w-full rounded-borders shadow-2')
                        with ui.column().classes('col-5'):
                            with ui.card().classes('w-full h-full tutorial-card'):
                                with ui.card_section():
                                    ui.markdown("#### 4. Collision Rule"); ui.markdown(f"While **carrying a <span style='color: {BLOCK_COLOR_1};'>block</span>**, the <span style='color: {AGENT_COLOR};'>**agent**</span> **cannot** move through other <span style='color: {BLOCK_COLOR_2};'>**blocks**</span>.")
                                with ui.card_section().classes('q-pt-none'): ui.video(f'/{TUTORIAL_DATA_ROOT}/tutorial/collision.mp4', autoplay=True, loop=True, muted=True).classes('w-full rounded-borders shadow-2')
                ui.separator()
                with ui.card_section():
                    ui.markdown("### Your Task: Predicting and Describing"); ui.markdown("After observing examples, you will be asked to do two things:")
                    with ui.list():
                        with ui.item():
                            with ui.item_section(): ui.item_label("1. Predict the Agent's Next 5 Moves").classes('text-bold'); ui.item_label("You will use the action buttons to enter the agent's next five actions. If you make a mistake, you can use the 'Undo' button."); ui.chip('Undo', icon='undo').props('color=warning text-color=white q-mt-xs')
                        with ui.item():
                            with ui.item_section(): ui.item_label("2. Describe the Agent's Behavior").classes('text-bold'); ui.item_label("Your main goal is to figure out the agent's strategy. A good description captures the rules it seems to be following.")
                    ui.markdown(f"> Example Description: The agent always moves to the closest green block and carries it to the top-left corner.")
            ui.markdown("Click 'Continue' when you are ready to begin the first task.").classes('text-center q-mt-lg')
            with ui.row().classes('w-full justify-center q-mt-md'):
                continue_button = ui.button("Continue", on_click=lambda: stage.set_user_data(finished=True)).props('color=primary size=lg')
        while not stage.get_user_data("finished", False): await asyncio.sleep(0.1)
    return nicewebrl.FeedbackStage(name="Experiment Tutorial", display_fn=tutorial_display_fn, user_save_file_fn=get_user_save_file_fn, metadata={})

def create_prediction_stages():
    stages = []; num_tasks = 10; num_videos_to_select = 4
    if not DATA_LOADED: return []
    num_total_variants = len(all_trajectories)
    for task_idx in range(num_tasks):
        valid_variants = [i for i in range(num_total_variants) if all_trajectories[i][task_idx] is not None]
        if len(valid_variants) < num_videos_to_select:
            logger.warning(f"Task {task_idx} has only {len(valid_variants)} valid videos. Need {num_videos_to_select}. Skipping this task.")
            continue
        chosen_variant_indices = random.sample(valid_variants, num_videos_to_select)
        all_possible_files = [f'video_{j}_{task_idx}' for j in chosen_variant_indices]
        demo_files = all_possible_files[:3]; predict_file = all_possible_files[3]
        metadata = {'demo_files': demo_files, 'predict_file': predict_file, 'task': task_list[task_idx], 'task_idx': task_idx}
        prediction_stage = nicewebrl.FeedbackStage(name=f"", display_fn=prediction_display_fn_with_state_loading, user_save_file_fn=get_user_save_file_fn, metadata=metadata)
        stages.append(prediction_stage)
    return stages

tutorial_stage = create_tutorial_stage()
all_stages = [tutorial_stage] + create_prediction_stages()

def create_experiment():
    return nicewebrl.Experiment(stages=all_stages, name="Human Gameplay Prediction Study", description="Multi-step prediction and rating of human gameplay behavior")

if __name__ == "__main__":
    ui.add_static_files(f'/{HUMAN_DATA_ROOT}', HUMAN_DATA_ROOT)
    ui.add_static_files(f'/{TUTORIAL_DATA_ROOT}', TUTORIAL_DATA_ROOT)
    experiment = create_experiment()
    app.title = "Human Gameplay Prediction Study"
    nicewebrl.run_experiment(experiment, port=8080, reload=False, show=True)