from typing import Optional, cast

import jax
import jax.numpy as jnp
import nicewebrl
from flax import struct
from nicegui import ui, app
import os
from nicewebrl import (
    EnvStage,
    JaxWebEnv,
    Stage,
    TimeStep,
    TimestepWrapper,
    base64_npimage,
    get_logger,
    Block,
    prepare_blocks,
    generate_stage_order,
)

from environment_jax import AutomaticityEnv, State, state_to_image_jit

logger = get_logger(__name__)

MAX_STAGE_EPISODES = 1
MAX_EPISODE_TIMESTEPS = 30
MIN_SUCCESS_EPISODES = 1
VERBOSITY = 1

NAME = os.environ.get('NAME', 'exp')
DEBUG = int(os.environ.get('DEBUG', 0))
DATA_DIR = os.environ.get('DATA_DIR', 'data')

def get_user_save_file_fn():
    return f'{DATA_DIR}/user={app.storage.user.get("seed")}_name={NAME}_debug={DEBUG}.json'


########################################
# Define Automotacity Env
########################################
# make environment
grid_size = 7
tile_size = 12
env = AutomaticityEnv(
    num_agents=1,
    size=grid_size,
    max_steps=MAX_EPISODE_TIMESTEPS,
    num_blocks=7,
    num_walls=3,
)


class AutomaticityEnvAdaptor:
    def __init__(self, env: AutomaticityEnv):
        self._env = env
        self.actions = env.actions
        self.default_params = None

    def step(self, _key, state, action):
        obs, state = self._env.step(state, action)
        reward = jnp.float32(0.0)
        done = False
        info = {}
        return obs, state, reward, done, info

    def reset(self, key):
        return self._env.reset(key)


jax_env = AutomaticityEnvAdaptor(env)
########################################
# Define actions and corresponding keys
########################################
actions = jnp.arange(len(jax_env.actions)).reshape(-1, 1)
action_array = actions
# noop, right, left, down, up, interact
action_keys = ["n", "ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", " "]
action_to_name = ["NOOP", "Right", "Left", "Down", "Up", "Pick up/Drop block"]

# NiceWebRL exploits a `TimeStep` object for checking episode conditions
# wrap environment in wrapper if needed
dummy_env_params = None
jax_env = TimestepWrapper(jax_env, autoreset=False, use_params=False)

# create web environment wrapper
jax_web_env = JaxWebEnv(env=jax_env, actions=action_array)

# Call this function to pre-compile jax functions before experiment starts.
jax_web_env.precompile(dummy_env_params=dummy_env_params)


# Define rendering function
def render_fn(timestep: nicewebrl.TimeStep):
    image = state_to_image_jit(timestep.observation[0], grid_size, tile_size)
    image = jnp.repeat(jnp.repeat(image, repeats=2, axis=0), repeats=2, axis=1)
    return image.astype(jnp.uint8)


# jit it so fast
render_fn = jax.jit(render_fn)

# precompile vmapped render fn that will vmap over all actions
vmap_render_fn = jax_web_env.precompile_vmap_render_fn(
    render_fn, jax_env.default_params
)


########################################
# Define Stages of experiment
########################################
all_stages = []
all_blocks = []


# ------------------
# Instruction stage
# ------------------
async def instruction_display_fn(stage, container):
    with container.style("align-items: center;"):
        nicewebrl.clear_element(container)
        ui.markdown(f"## {stage.name}").style("font-size: 2em; text-align: center;")
        ui.markdown(
            """
            # 👋 Welcome! 
            In this game, you'll control a red circular agent in a grid world.

            Throughout the experiment, you'll be given different tasks to complete.
            
            ## For each task:
            - 📝 Follow the instructions to the best of your understanding
            - 🔄 Keep repeating the behavior until the time limit
            - ✨ Don't worry if you can't finish before time runs out!

            ## Controls:
            * 🎮 Arrow keys ↑ ↓ ← → - Move the agent
            * 🔲 Spacebar - Interact with objects  
            * ⏸️ 'N' key - Stay in place
            """
        ).style("font-size: 1.2em; text-align: center;")


instruction_stage = Stage(name="Instuctions", display_fn=instruction_display_fn)
all_stages.append(instruction_stage)


# ------------------
# Environment stage
# ------------------
# EXAMPLE: change parameters for this specific stage
# env_params = jax_env.default_params.replace(
#     max_timesteps=MAX_EPISODE_TIMESTEPS,
# )


def make_image_html(src):
    html = f"""
  <div id="stateImageContainer" style="display: flex; justify-content: center; align-items: center;">
      <img id="stateImage" src="{src}" style="width: 100%; height: 100%; object-fit: contain;">
  </div>
  """
    return html


async def env_stage_display_fn(
    stage: EnvStage, container: ui.element, timestep: TimeStep
):
    state_image = stage.render_fn(timestep)
    state_image = base64_npimage(state_image)
    stage_state = stage.get_user_data("stage_state")
    metadata = stage.metadata

    with container.style("align-items: center;"):
        nicewebrl.clear_element(container)
        # --------------------------------
        # tell person how many episodes completed and how many successful
        # --------------------------------
        with ui.row():
            with ui.element("div").classes("p-2 bg-green-100 text-center").style("margin: 0 auto;"):
                ui.label().bind_text_from(
                    metadata, "task", lambda task: f"Task: {task}"
                ).style("font-size: 1.2em; font-weight: bold; text-align: center;")
            with ui.element("div").classes("p-2 bg-blue-100 text-center").style("margin: 0 auto;"):
                ui.label(
                    f"Press the arrow keys to move the agent (up, down, left, right).\nPress the space bar to pick up/drop a block. Press the letter 'n' to stay in place."
                ).style("font-size: 1.0em; justify-content: center;")

        # --------------------------------
        # display environment
        # --------------------------------
        ui.html(make_image_html(src=state_image)).style("width: 75%")
        

def evaluate_success_fn(timestep: TimeStep, params: Optional[struct.PyTreeNode] = None):
    """Episode finishes if person gets 5 achievements"""
    # achievements = timestep.state.achievements.astype(jnp.float32)
    # success = achievements.sum() >= 5
    return cast(State, timestep.state).terminal



# possible tasks
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

async def task_display_fn(stage, container):
    task = stage.metadata['task']
    with container.style("align-items: center;"):
        nicewebrl.clear_element(container)
        
        # Title with task
        ui.markdown(f"# {task}").style(
            "font-size: 2em; text-align: center; color: #2563eb; margin: 20px 0;"
        )

        # Controls section
        with ui.card().style("width: 80%; margin: 20px auto; text-align: center;"):
            ui.markdown("## Controls").style("font-size: 1.5em; color: #1f2937; text-align: center; margin: 0 auto;")
            
            with ui.column().style("gap: 15px; align-items: center;"):
                ui.markdown("🎮 **Movement**: Use arrow keys: ⬆️ ⬇️ ⬅️ ➡️").style("font-size: 1.2em;")
                
                ui.markdown("🔲 **Block Interaction**: Press **[Space]** to pick up/drop blocks. *(You can only pick up blocks you're standing on)*").style("font-size: 1.2em;")
                
                ui.markdown("⏸️ **Stay in Place**: Press **[N]** to not move").style("font-size: 1.2em;")



tutorial_stage = Stage(
        name="Tutorial",
        title="Tutorial: You will now have a chance to practice the task. Try moving the agent around the grid and picking up/dropping blocks.",
        body="You will now have a chance to practice the task. Try moving the agent around the grid and picking up/dropping blocks.",
        metadata={'task': "Tutorial"},
        display_fn=task_display_fn
    )
all_stages.append(tutorial_stage)
tutorial_env_stage = EnvStage(
    user_save_file_fn=get_user_save_file_fn,
    name="Environment",
    web_env=jax_web_env,
    action_keys=action_keys,
    action_to_name=action_to_name,
    env_params=None,
    render_fn=render_fn,
    vmap_render_fn=vmap_render_fn,
    display_fn=env_stage_display_fn,
    evaluate_success_fn=evaluate_success_fn,
    min_success=MIN_SUCCESS_EPISODES,
    max_episodes=MAX_STAGE_EPISODES,
    verbosity=VERBOSITY,
    metadata={'task': "Tutorial: You will now have a chance to practice the task. Try moving the agent around the grid and picking up/dropping blocks."},    # add custom metadata to be stored here
)
all_stages.append(tutorial_env_stage)

for t_id, t in enumerate(task_list):

    instruct_stage = Stage(
        name=f"New Task",
        title=t,
        body=f"Complete the task: {t}",
        metadata={'task': t},
        display_fn=task_display_fn
    )
    all_stages.append(instruct_stage)

    environment_stage = EnvStage(
        user_save_file_fn=get_user_save_file_fn,
        name="Environment",
        web_env=jax_web_env,
        action_keys=action_keys,
        action_to_name=action_to_name,
        env_params=None,
        render_fn=render_fn,
        vmap_render_fn=vmap_render_fn,
        display_fn=env_stage_display_fn,
        evaluate_success_fn=evaluate_success_fn,
        min_success=MIN_SUCCESS_EPISODES,
        max_episodes=MAX_STAGE_EPISODES,
        verbosity=VERBOSITY,
        metadata={'task': t},    # add custom metadata to be stored here
    )

    all_stages.append(environment_stage)