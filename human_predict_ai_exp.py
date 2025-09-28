import asyncio
import os
import pickle
import uuid
from datetime import datetime
from typing import List, Optional

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
USE_FSM_DATA = True
PREDICTION_STEPS = 5

logger = get_logger(__name__)

NAME = os.environ.get('NAME', 'prediction_exp_method1')
DEBUG = int(os.environ.get('DEBUG', 0))
DATA_DIR = os.environ.get('DATA_DIR', 'human_pred_data_curated_method1')
FSM_DATA_ROOT = 'human_vids_curated'

task_list = [
    "block_cycle", "clockwise_patrol", "counter_patrol", "left_right",
    "pair_blue", "patrol_with_a_star", "pattern_l", "pickup_green_a_star",
    "snake", "up_down",
]

action_to_name = ["NOOP", "Right", "Left", "Down", "Up", "Pick up/Drop block"]

# ==============================================================================
#                               DATA LOADING
# ==============================================================================
def load_fsm_gameplay_data(data_path: str):
    logger.info(f"Attempting to load FSM gameplay data from {data_path}...")
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        logger.info("Successfully loaded FSM data from pickle file.")
        return data
    except Exception as e:
        logger.error(f"Could not load or parse FSM gameplay data: {e}", exc_info=True)
        return None

def get_trajectory_data(all_trajectories, video_file_prefix):
    try:
        parts = video_file_prefix.split('_')
        variant_idx = int(parts[1])
        task_idx = int(parts[2])
        return all_trajectories[variant_idx][task_idx]
    except (IndexError, ValueError, KeyError) as e:
        logger.error(f"Could not find data for '{video_file_prefix}': {e}")
        return None

def get_ground_truth_actions(trajectory_data):
    try:
        actions_array = trajectory_data['actions']
        start_idx = 1
        end_idx = start_idx + PREDICTION_STEPS
        return actions_array[start_idx:end_idx]
    except Exception as e:
        logger.error(f"Could not extract ground truth actions: {e}")
        return None

DATA_LOADED = False
try:
    if USE_FSM_DATA:
        FSM_DATA_FILE = os.path.join(FSM_DATA_ROOT, 'fsm_curated_gameplay_data.pkl')
        all_trajectories = load_fsm_gameplay_data(FSM_DATA_FILE)
    else:
        raise NotImplementedError("Human gameplay data loading not implemented in this version.")
    if all_trajectories is not None:
        DATA_LOADED = True
    else:
        raise ValueError("Data loading returned None.")
except Exception as e:
    logger.error(f"FATAL: Could not load gameplay data: {e}")
    all_trajectories = None
    DATA_LOADED = False

# ==============================================================================
#                               ENVIRONMENT SETUP
# ==============================================================================
def setup_prediction_environment():
    GRID_SIZE = 7
    TILE_SIZE = 12
    return AutomaticityEnv(num_agents=1, size=GRID_SIZE, max_steps=30, num_blocks=10, num_walls=30), GRID_SIZE, TILE_SIZE

prediction_env, GRID_SIZE, TILE_SIZE = setup_prediction_environment()

def load_prediction_data(video_file_prefix):
    if not DATA_LOADED:
        _, state_t0 = prediction_env.reset(jax.random.PRNGKey(hash(video_file_prefix)))
        _, state_t1 = prediction_env.step(state_t0, jnp.array([0]))
        return state_t0, state_t1, state_t1, None

    trajectory_data = get_trajectory_data(all_trajectories, video_file_prefix)
    if trajectory_data is None:
        _, state_t0 = prediction_env.reset(jax.random.PRNGKey(hash(video_file_prefix)))
        _, state_t1 = prediction_env.step(state_t0, jnp.array([0]))
        return state_t0, state_t1, state_t1, None

    try:
        states_pytree = trajectory_data['states']
        state_t0 = jax.tree.map(lambda x: x[0], states_pytree)
        state_t1 = jax.tree.map(lambda x: x[1], states_pytree)
        ground_truth = get_ground_truth_actions(trajectory_data)
        return state_t0, state_t1, state_t1, ground_truth
    except Exception as e:
        logger.error(f"Failed to load prediction start frames for {video_file_prefix}: {e}")
        _, state_t0 = prediction_env.reset(jax.random.PRNGKey(hash(video_file_prefix)))
        _, state_t1 = prediction_env.step(state_t0, jnp.array([0]))
        return state_t0, state_t1, state_t1, None

# ==============================================================================
#                           DATA STORAGE FUNCTIONS
# ==============================================================================
def store_stage_response(stage_idx, stage_data):
    if "all_stage_responses" not in app.storage.user:
        app.storage.user["all_stage_responses"] = []
    stage_response = {'stage_idx': stage_idx, 'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"), **stage_data}
    app.storage.user["all_stage_responses"].append(stage_response)
    for key, value in stage_data.items():
        app.storage.user[key] = value

def get_user_save_file_fn():
    if "user_id" not in app.storage.user:
        app.storage.user["user_id"] = str(uuid.uuid4())[:8]
    if "session_timestamp" not in app.storage.user:
        app.storage.user["session_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'predict_user={app.storage.user["user_id"]}_session={app.storage.user["session_timestamp"]}_name={NAME}_debug={DEBUG}.json'
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)

# ==============================================================================
#                       MULTI-STEP PREDICTION INTERFACE
# ==============================================================================
async def multistep_prediction_interface(container, initial_state, ground_truth_actions, num_steps=5):
    nicewebrl.clear_element(container)
    
    state = {
        "current_step": 0,
        "predicted_actions": [],
        "gt_state_history": [initial_state],
        "current_gt_state": initial_state,
        "preview_action": None,
        "last_human_action": "N/A",
        "last_gt_action": "N/A",
    }
    completion_event = asyncio.Event()

    def _draw_arrow(image, pos1, pos2, color):
        cx1, cy1 = int(pos1[0] * TILE_SIZE + TILE_SIZE / 2), int(pos1[1] * TILE_SIZE + TILE_SIZE / 2)
        cx2, cy2 = int(pos2[0] * TILE_SIZE + TILE_SIZE / 2), int(pos2[1] * TILE_SIZE + TILE_SIZE / 2)
        num_points = TILE_SIZE * 2
        for j in range(num_points + 1):
            alpha = j / num_points
            px, py = int(cx1 * (1 - alpha) + cx2 * alpha), int(cy1 * (1 - alpha) + cy2 * alpha)
            if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                image = image.at[py, px].set(color)
        vec_x, vec_y = cx2 - cx1, cy2 - cy1
        norm = np.sqrt(vec_x**2 + vec_y**2)
        if norm > 0:
            ux, uy = vec_x / norm, vec_y / norm
            arrow_len = TILE_SIZE / 3
            p1x, p1y = int(cx2 - ux * arrow_len + uy * arrow_len / 2), int(cy2 - uy * arrow_len - ux * arrow_len / 2)
            p2x, p2y = int(cx2 - ux * arrow_len - uy * arrow_len / 2), int(cy2 - uy * arrow_len + ux * arrow_len / 2)
            for j in range(int(arrow_len) + 1):
                alpha = j / int(arrow_len)
                apx1, apy1 = int(p1x * (1 - alpha) + cx2 * alpha), int(p1y * (1 - alpha) + cy2 * alpha)
                if 0 <= apy1 < image.shape[0] and 0 <= apx1 < image.shape[1]:
                    image = image.at[apy1, apx1].set(color)
                apx2, apy2 = int(p2x * (1 - alpha) + cx2 * alpha), int(p2y * (1 - alpha) + cy2 * alpha)
                if 0 <= apy2 < image.shape[0] and 0 <= apx2 < image.shape[1]:
                    image = image.at[apy2, apx2].set(color)
        return image

    def render_state_with_trajectories(base_state: State, preview_action_idx: Optional[int] = None):
        try:
            obs = prediction_env.get_observation(base_state)[0]
            base_image = state_to_image_jit(obs, GRID_SIZE, TILE_SIZE)
            enhanced_image = jnp.array(base_image, dtype=jnp.float32)
            gt_positions = [tuple(s.agent_locations[0].tolist()) for s in state["gt_state_history"]]
            if len(gt_positions) > 1:
                gt_line_color = jnp.array([200, 200, 0], dtype=jnp.float32)
                for i in range(len(gt_positions) - 1):
                    x, y = int(gt_positions[i][0]), int(gt_positions[i][1])
                    for dy in range(TILE_SIZE):
                        for dx in range(TILE_SIZE):
                            py, px = (y * TILE_SIZE) + dy, (x * TILE_SIZE) + dx
                            enhanced_image = enhanced_image.at[py, px, 0].set(jnp.clip(enhanced_image[py, px, 0] * 0.7 + 60, 0, 255))
                            enhanced_image = enhanced_image.at[py, px, 1].set(jnp.clip(enhanced_image[py, px, 1] * 0.6, 0, 255))
                            enhanced_image = enhanced_image.at[py, px, 2].set(jnp.clip(enhanced_image[py, px, 2] * 0.7 + 120, 0, 255))
                    enhanced_image = _draw_arrow(enhanced_image, gt_positions[i], gt_positions[i+1], gt_line_color)
            if preview_action_idx is not None:
                _, preview_next_state = prediction_env.step(base_state, jnp.array([preview_action_idx]))
                preview_line_color = jnp.array([0, 150, 255], dtype=jnp.float32)
                enhanced_image = _draw_arrow(enhanced_image, gt_positions[-1], tuple(preview_next_state.agent_locations[0].tolist()), preview_line_color)
            final_image = enhanced_image.astype(jnp.uint8)
            scaled_image = jnp.repeat(jnp.repeat(final_image, 3, 0), 3, 1)
            return nicewebrl.base64_npimage(scaled_image)
        except Exception as e:
            logger.error(f"Error rendering trajectory: {e}")
            return None

    with container.style("align-items: center;"):
        progress_container = ui.column().classes('items-center w-full max-w-lg mb-4')
        env_container = ui.column().style("align-items: center;")
        button_container = ui.column().style("align-items: center;")

        def update_all_displays():
            nicewebrl.clear_element(progress_container)
            with progress_container:
                with ui.column().classes('w-full items-center gap-0'):
                    ui.label(f"Step {state['current_step'] + 1} of {num_steps}").classes('text-xl font-light text-gray-500')
                    ui.linear_progress(value=state['current_step'] / num_steps, size='15px').props('color=primary rounded')
                    if state['current_step'] > 0:
                        ui.separator().classes('w-3/4 my-2')
                        with ui.row().classes('w-full justify-around items-center'):
                            with ui.column().classes('items-center gap-0'):
                                ui.label('Your Last Prediction').classes('text-xs uppercase font-medium text-gray-400')
                                ui.label(state["last_human_action"]).classes('text-lg font-bold')
                            with ui.column().classes('items-center gap-0'):
                                ui.label("Agent's Actual Action").classes('text-xs uppercase font-medium text-gray-400')
                                ui.label(state['last_gt_action']).classes('text-lg font-bold')
            
            nicewebrl.clear_element(env_container)
            with env_container:
                img_b64 = render_state_with_trajectories(state["current_gt_state"], state["preview_action"])
                if img_b64:
                    ui.html(f'<img src="{img_b64}" style="width: 450px; height: auto; border: 2px solid #666; border-radius: 4px;">')
                    if len(state["gt_state_history"]) > 1 or state["preview_action"] is not None:
                        with ui.column().classes('w-full items-center'):
                            ui.html("""
                                <div style="margin-top: 5px; margin-bottom: 0; padding: 8px; background: #f0f0f0; border-radius: 6px; font-size: 13px; text-align: center;">
                                    <span style="color: #9575cd;">■</span> Visited&nbsp;&nbsp;
                                    <b style="color: #b2a300; vertical-align: middle;">→</b> Agent's Path&nbsp;&nbsp;
                                    <b style="color: #0096FF; vertical-align: middle;">→</b> Your Preview
                                </div>
                            """)
            nicewebrl.clear_element(button_container)
            with button_container:
                if state["preview_action"] is not None:
                    action_name = action_to_name[state["preview_action"]]
                    ui.markdown(f"You selected **{action_name}**. Click the **{action_name}** button again to confirm, or choose a different action.").classes('text-center p-2 mt-2 bg-blue-100 border border-blue-300 rounded-md')
                else:
                    ui.markdown("**Choose the agent's next action:**").classes("mt-2")
                
                button_configs = [
                    {'label': '↑ Up', 'action_idx': 4, 'color': 'primary'},
                    {'label': '← Left', 'action_idx': 2, 'color': 'primary'},
                    {'label': '↓ Down', 'action_idx': 3, 'color': 'primary'},
                    {'label': '→ Right', 'action_idx': 1, 'color': 'primary'},
                    {'label': '⏹ Stay', 'action_idx': 0, 'color': 'secondary'},
                    {'label': '🔄 Pick up/Drop', 'action_idx': 5, 'color': 'accent'},
                ]
                with ui.column(align_items='center').classes('mt-2 q-gutter-y-sm'):
                    with ui.row().classes('q-gutter-x-sm'):
                        btn = button_configs[0]
                        props = 'size=lg' if state["preview_action"] == btn['action_idx'] else 'size=lg outline'
                        ui.button(btn['label'], on_click=lambda b=btn: handle_action(b['action_idx'])).props(f"color={btn['color']} {props}")
                    with ui.row().classes('q-gutter-x-sm'):
                        for btn in button_configs[1:4]:
                            props = 'size=lg' if state["preview_action"] == btn['action_idx'] else 'size=lg outline'
                            ui.button(btn['label'], on_click=lambda b=btn: handle_action(b['action_idx'])).props(f"color={btn['color']} {props}")
                    with ui.row().classes('q-gutter-x-sm'):
                        for btn in button_configs[4:]:
                             props = '' if state["preview_action"] == btn['action_idx'] else 'outline'
                             ui.button(btn['label'], on_click=lambda b=btn: handle_action(b['action_idx'])).props(f"color={btn['color']} {props}")
        
        async def handle_action(clicked_action_idx):
            if state["preview_action"] == clicked_action_idx:
                state["predicted_actions"].append(clicked_action_idx)
                gt_action = ground_truth_actions[state["current_step"]]
                state["last_human_action"] = action_to_name[clicked_action_idx]
                state["last_gt_action"] = action_to_name[gt_action]
                _, new_state = prediction_env.step(state["current_gt_state"], jnp.array([gt_action]))
                state["current_gt_state"] = new_state
                state["gt_state_history"].append(new_state)
                state["preview_action"] = None
                state["current_step"] += 1
                if state["current_step"] >= num_steps:
                    await finish_prediction()
                else:
                    update_all_displays()
            else:
                state["preview_action"] = clicked_action_idx
                update_all_displays()

        async def finish_prediction():
            nicewebrl.clear_element(progress_container)
            nicewebrl.clear_element(button_container)
            with env_container:
                nicewebrl.clear_element(env_container)
                img_b64 = render_state_with_trajectories(state["current_gt_state"])
                if img_b64:
                    ui.html(f'<img src="{img_b64}" style="width: 450px; height: auto; border: 2px solid #666; border-radius: 4px;">')
            with button_container:
                ui.markdown("## ✅ Prediction Round Complete!").classes("mt-4")
                ui.button("✓ Continue", on_click=completion_event.set).props("color=primary size=lg")

        update_all_displays()
        
        await completion_event.wait()
        return state["predicted_actions"]
        
async def prediction_display_fn_with_state_loading(stage, container):
    ui.run_javascript("window.scrollTo(0, 0); document.body.scrollTop = 0; document.documentElement.scrollTop = 0;")
    nicewebrl.clear_element(container)
    demo_files = stage.metadata['demo_files']
    predict_file = stage.metadata['predict_file']

    with container.style("align-items: center; width: 100%;"):
        ui.markdown(f"## {stage.name}")
        ui.markdown("### Part 1: Observe Agent Behavior")
        ui.markdown("Watch the following **three examples** of the agent's behavior. The agent is performing the same underlying task in each video, but the environment is different. **You can replay the video as many times as you want.**")
        with ui.row().classes('w-full no-wrap justify-center items-start').style("gap: 15px; margin-top: 20px;"):
            for i, file_prefix in enumerate(demo_files):
                with ui.column().style("align-items: center; max-width: 280px;"):
                    ui.markdown(f"**Example {i+1}**")
                    demo_video_path = f"/{FSM_DATA_ROOT}/{file_prefix}.mp4"
                    ui.video(demo_video_path, autoplay=False, muted=True, controls=True).style("width: 280px;")

        demos_watched = asyncio.Event()
        ui.button("I have watched the examples. Proceed to the prediction task.", on_click=lambda: (demos_watched.set(), ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)"))).props("color=primary size=lg").style("margin-top: 25px;")
        await demos_watched.wait()

        ui.markdown("---").style("width: 80%; margin: 30px auto;")
        ui.markdown("### Part 2: Predict the Agent's Next Moves")
        ui.markdown(f"Now, you will see the starting state and the next frame for a new trial. Based on what you learned, predict the agent's next **{PREDICTION_STEPS} moves**, one by one.")

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
                ui.markdown("**Your Starting Point (Time=1)**")
                obs_t1 = prediction_env.get_observation(state_t1)[0]
                img_t1 = state_to_image_jit(obs_t1, GRID_SIZE, TILE_SIZE)
                scaled_img_t1 = np.repeat(np.repeat(img_t1, 3, 0), 3, 1)
                ui.image(nicewebrl.base64_npimage(scaled_img_t1)).style("border: 2px solid #333; border-radius: 4px; width: 250px;")

        prediction_started = asyncio.Event()
        ui.button("Start Prediction", on_click=lambda: (prediction_started.set(), ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)"))).props("color=positive size=lg")
        await prediction_started.wait()

        prediction_container = ui.column().style("width: 100%;")
        predicted_actions = await multistep_prediction_interface(container=prediction_container, initial_state=initial_prediction_state, ground_truth_actions=ground_truth_actions, num_steps=PREDICTION_STEPS)
    
    nicewebrl.clear_element(container)
    ui.run_javascript("window.scrollTo(0, 0);")

    with container.style("align-items: center; width: 100%;"):
        ui.markdown("### Part 3: Rate the Agent's Behavior")
        ui.markdown("*Please answer the following questions about the agent's behavior based on all the videos you've seen for this task.*")
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
            gt_actions_list = ground_truth_actions.tolist() if ground_truth_actions is not None else []
            stage_data = {"predicted_actions_sequence": predicted_actions, "ground_truth_actions": gt_actions_list, "demo_video_files": demo_files, "prediction_video_file": predict_file, "chosen_video_file": predict_file, "behavior_description": behavior_description.value, "task_idx": stage.metadata.get('task_idx'), "task_name": stage.metadata.get('task'), **{f"{name}_rating": r["value"] for name, r in ratings.items()}}
            store_stage_response(stage.metadata.get('task_idx', -1), stage_data)
            success = True
        submit_button = ui.button("Submit and Continue to Next Task", on_click=submit).props("color=primary size=lg")
        while not success:
            await submit_button.clicked()

    await stage.set_user_data(finished=True)

# ==============================================================================
#                           EXPERIMENT CREATION
# ==============================================================================
def create_tutorial_stage():
    """Creates a final, polished tutorial page with a 2x2 grid and detailed UI explanations."""
    async def tutorial_display_fn(stage, container):
        nicewebrl.clear_element(container)
        ui.run_javascript("window.scrollTo(0, 0);")

        AGENT_COLOR = "#D32F2F"    # Red for the agent
        BLOCK_COLOR_1 = "#32D219"  # Green for blocks
        BLOCK_COLOR_2 = "#1976D2"  # Blue for blocks
        WALL_COLOR = "#616161"     # Gray for walls
        
        with container.style("max-width: 1000px; margin: auto; padding: 20px;"):
            with ui.card().classes('w-full'):
                with ui.card_section():
                    ui.markdown("## Welcome to the Experiment!")
                    ui.markdown("In this experiment, you will observe an agent's behavior to understand its goals. First, let's learn the rules of its world and your task.")
                
                ui.separator()

                with ui.card_section():
                    ui.html('<style>.tutorial-card { min-height: 750px; }</style>')
                    with ui.row().classes('w-full q-gutter-md'):
                        # Rule 1: Movement
                        with ui.column().classes('col-5'):
                            with ui.card().classes('w-full h-full tutorial-card'):
                                with ui.card_section():
                                    ui.markdown("#### 1. Basic Movement")
                                    ui.markdown(f"The <span style='color: {AGENT_COLOR};'>**agent**</span> moves one square at a time using the action buttons.")
                                    with ui.row().classes('q-gutter-sm q-mt-xs'):
                                        ui.chip('Up', icon='arrow_upward')
                                        ui.chip('Down', icon='arrow_downward')
                                        ui.chip('Left', icon='arrow_back')
                                        ui.chip('Right', icon='arrow_forward')
                                        ui.chip('Stay', icon='pause')
                                with ui.card_section().classes('q-pt-none'):
                                    ui.video('/human_vids_curated/tutorial/movement.mp4', autoplay=True, loop=True, muted=True).classes('w-full rounded-borders shadow-2')

                        # Rule 2: Walls
                        with ui.column().classes('col-5'):
                            with ui.card().classes('w-full h-full tutorial-card'):
                                with ui.card_section():
                                    ui.markdown("#### 2. Walls are Obstacles")
                                    ui.markdown(f"The <span style='color: {AGENT_COLOR};'>**agent**</span> cannot move through solid gray <span style='color: {WALL_COLOR};'>**walls**</span>.")
                                with ui.card_section().classes('q-pt-none'):
                                    ui.video('/human_vids_curated/tutorial/walls.mp4', autoplay=True, loop=True, muted=True).classes('w-full rounded-borders shadow-2')

                    with ui.row().classes('w-full q-gutter-md q-mt-md'):
                        # Rule 3: Blocks
                        with ui.column().classes('col-5'):
                            with ui.card().classes('w-full h-full tutorial-card'):
                                with ui.card_section():
                                    ui.markdown("#### 3. Picking Up & Dropping Blocks")
                                    ui.markdown(f"To **Pick Up** a <span style='color: {BLOCK_COLOR_1};'>**block**</span>, the <span style='color: {AGENT_COLOR};'>**agent**</span> must first move onto its square, then use the 'Pick up/Drop' action.")
                                    ui.markdown("To **Drop** a held block, move to an empty square and use the same action again.")
                                    ui.chip('Pick up/Drop', icon='swap_horiz').props('color=accent text-color=white q-mt-sm')
                                    ui.markdown(f"<br><b>Visual Indicator:</b> When the agent is carrying a block, a small colored square appears in its center.")
                                with ui.card_section().classes('q-pt-none'):
                                    ui.video('/human_vids_curated/tutorial/pickup_drop.mp4', autoplay=True, loop=True, muted=True).classes('w-full rounded-borders shadow-2')

                        # Rule 4: Collision
                        with ui.column().classes('col-5'):
                            with ui.card().classes('w-full h-full tutorial-card'):
                                with ui.card_section():
                                    ui.markdown("#### 4. Collision Rule")
                                    ui.markdown(f"While **carrying a <span style='color: {BLOCK_COLOR_1};'>block</span>**, the agent **cannot** move through other <span style='color: {BLOCK_COLOR_2};'>**blocks**</span>.")
                                with ui.card_section().classes('q-pt-none'):
                                    ui.video('/human_vids_curated/tutorial/collision.mp4', autoplay=True, loop=True, muted=True).classes('w-full rounded-borders shadow-2')
                ui.separator()

                with ui.card_section():
                    ui.markdown("### Your Task: Predicting and Describing")
                    ui.markdown("After observing examples, you will be asked to do two things:")
                    with ui.list():
                        with ui.item():
                            with ui.item_section():
                                ui.item_label("1. Predict the Agent's Next 5 Moves").classes('text-bold')
                                ui.item_label("You will predict the agent's next five actions one by one. You will click an action once to see a preview, and then click the same action again to confirm your choice.")
                        with ui.item():
                            with ui.item_section():
                                ui.item_label("2. Describe the Agent's Behavior").classes('text-bold')
                                ui.item_label("Your main goal is to figure out the agent's strategy. A good description captures the rules it seems to be following.")
                    ui.markdown(f"> *Example Description: The agent always moves to the closest green block and carries it to the top-left corner.*")

            with ui.row().classes('w-full justify-center q-mt-md'):
                ui.button("Continue", on_click=lambda: stage.set_user_data(finished=True)).props('color=primary size=lg')
            
        while not stage.get_user_data("finished", False):
            await asyncio.sleep(0.1)

    return nicewebrl.FeedbackStage(
        name="Experiment Tutorial",
        display_fn=tutorial_display_fn,
        user_save_file_fn=get_user_save_file_fn,
        metadata={}
    )

def create_prediction_stages() -> List[nicewebrl.FeedbackStage]:
    stages = []
    num_tasks = 10
    num_videos_per_task = 4
    for task_idx in range(num_tasks):
        all_possible_files = [f'video_{j}_{task_idx}' for j in range(num_videos_per_task)]
        demo_files, predict_file = all_possible_files[:3], all_possible_files[3]
        metadata = {'demo_files': demo_files, 'predict_file': predict_file, 'task': task_list[task_idx], 'task_idx': task_idx}
        prediction_stage = nicewebrl.FeedbackStage(name="", display_fn=prediction_display_fn_with_state_loading, user_save_file_fn=get_user_save_file_fn, metadata=metadata)
        stages.append(prediction_stage)
    return stages

tutorial_stage = create_tutorial_stage()
all_stages = [tutorial_stage] + create_prediction_stages()

def create_experiment():
    return nicewebrl.Experiment(stages=all_stages, name="Agent Behavior Prediction Study", description="Multi-step prediction and rating of AI agent behavior")