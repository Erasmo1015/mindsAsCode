import asyncio
import os.path
import struct
from asyncio import Lock

import aiofiles
import msgpack
import nicewebrl
from fastapi import Request
from nicegui import app, ui
from nicewebrl import stages
from nicewebrl.logging import get_logger, setup_logging
from tortoise import Tortoise
import os
import jax

import human_predict_human_exp as experiment


# Serve the directory with the human gameplay videos
app.add_static_files('/human_gameplay_videos', 'human_gameplay_videos')
# Also serve the old directory, because it contains the tutorial videos
app.add_static_files('/human_vids_curated', 'human_vids_curated')

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================
DATABASE_FILE = os.environ.get("DB_FILE", "db.sqlite")
# NEW: Default data dir for this experiment
DATA_DIR = os.environ.get("DATA_DIR", "human_gameplay_data")
DEBUG = int(os.environ.get("DEBUG", 0))
DEBUG_SEED = int(os.environ.get("SEED", 0))
NAME = os.environ.get("NAME", "exp_gameplay") # New experiment name
DATABASE_FILE = f"{DATABASE_FILE}_name={NAME}_debug={DEBUG}"

# Updated actions with clearer "Interact" description
ACTIONS = ["NOOP", "Right", "Left", "Down", "Up", "Pick up/Drop block"]
TOOLTIPS = [
    "Agent stays in place",
    "Agent moves right",
    "Agent moves left",
    "Agent moves down",
    "Agent moves up",
    "Agent picks up or drops a block"
]

os.makedirs(DATA_DIR, exist_ok=True)

_user_locks = {}

# ==============================================================================
#                               LOGGING SETUP
# ==============================================================================
def log_filename_fn(log_dir, user_id):
    return os.path.join(log_dir, f"log_{user_id}.log")

setup_logging(
    DATA_DIR,
    log_filename_fn=log_filename_fn,
    nicegui_storage_user_key="seed",
)
logger = get_logger("main")

# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================
def stage_name(stage):
    return stage.name

def get_user_lock():
    """A function that returns a lock for the current user using their unique seed"""
    user_seed = app.storage.user["seed"]
    if user_seed not in _user_locks:
        _user_locks[user_seed] = Lock()
    return _user_locks[user_seed]

async def experiment_not_finished():
    """Check if the experiment is not finished"""
    async with get_user_lock():
        stage_idx = app.storage.user.get("stage_idx", 0)
        not_finished = not app.storage.user.get("experiment_finished", False)
        not_finished &= stage_idx < len(experiment.all_stages)
    return not_finished

def blob_user_filename():
    """filename structure for user data in GCS (cloud)"""
    seed = app.storage.user["seed"]
    worker = app.storage.user.get("worker_id", None)
    if worker is not None:
        return f"user={seed}_worker={worker}_name={NAME}_debug={DEBUG}"
    else:
        return f"user={seed}_name={NAME}_debug={DEBUG}"

async def global_handle_key_press(e, container):
    """Define global key press handler"""
    stage_idx = app.storage.user["stage_idx"]
    if app.storage.user["stage_idx"] >= len(experiment.all_stages):
        return

    stage_order = app.storage.user["stage_order"]
    stage_id = stage_order[stage_idx]
    stage = experiment.all_stages[stage_id]
    if stage.get_user_data("finished", False):
        return

    await stage.handle_key_press(e, container)
    local_handle_key_press = stage.get_user_data("local_handle_key_press")
    if local_handle_key_press is not None:
        await local_handle_key_press()

async def save_data(final_save=True, feedback=None, **kwargs):
    """Save user data to file"""
    user_data_file = experiment.get_user_save_file_fn()

    if final_save:
        user_storage = nicewebrl.make_serializable(dict(app.storage.user))
        last_line = dict(
            finished=True,
            feedback=feedback,
            user_storage=user_storage,
            **kwargs,
        )
        async with aiofiles.open(user_data_file, "ab") as f:
            packed_data = msgpack.packb(last_line)
            await f.write(struct.pack(">i", len(packed_data)))
            await f.write(packed_data)

# ==============================================================================
#                               DATABASE SETUP
# ==============================================================================
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

async def init_db() -> None:
    await Tortoise.init(
        db_url=f"sqlite://{DATA_DIR}/{DATABASE_FILE}",
        modules={"models": ["models"]},
    )
    await Tortoise.generate_schemas()

async def close_db() -> None:
    await Tortoise.close_connections()

app.on_startup(init_db)
app.on_shutdown(close_db)

# ==============================================================================
#                       CONSENT FORM AND DEMOGRAPHICS
# ==============================================================================
async def make_consent_form(container):
    """Display consent form and wait for agreement"""
    consent_given = asyncio.Event()
    with container:
        ui.markdown("## Consent Form")
        with open("misc/consent.md", "r") as consent_file:
            consent_text = consent_file.read()
        ui.markdown(consent_text)
        ui.checkbox("I agree to participate.", on_change=lambda: consent_given.set())

    await consent_given.wait()

async def collect_demographic_info(container):
    """Collect basic demographic information"""
    nicewebrl.clear_element(container)
    with container:
        ui.markdown("## Demographic Information")
        ui.markdown("Please fill out the following information before finishing.")

        with ui.column().classes('items-center w-full'):
            age_input = ui.number("Age", min=1, max=100).style('width: 50%')
            ui.label("Sex")
            sex_input = ui.radio(["Male", "Female", "Non-binary"], value="Male").props("inline")

        success = False

        async def submit():
            nonlocal success
            age = age_input.value
            sex = sex_input.value

            if age is None:
                ui.notify("Please enter a valid age.", type="warning")
                return
            
            app.storage.user["age"] = int(age)
            app.storage.user["sex"] = sex
            logger.info(f"Demographics collected: age={int(age)}, sex={sex}")
            success = True

        button = ui.button("Submit", on_click=submit)
        while not success:
            await asyncio.sleep(0.1)

# ==============================================================================
#                               EXPERIMENT FLOW
# ==============================================================================
async def start_experiment(meta_container, stage_container, button_container):
    """Main experiment flow"""
    logger.info("STARTING EXPERIMENT")
    
    if not (app.storage.user.get("experiment_started", False) or DEBUG):
        await make_consent_form(stage_container)
        app.storage.user["experiment_started"] = True

    # Fullscreen handling
    if DEBUG == 0:
        ui.run_javascript("window.require_fullscreen = false")
    else:
        ui.run_javascript("window.require_fullscreen = false")

    # Register global key press handler
    ui.on("key", lambda e: global_handle_key_press(e, meta_container))

    # Run experiment stages
    logger.info("Starting experiment")
    while await experiment_not_finished():
        # Get current stage
        stage_idx = app.storage.user["stage_idx"]
        stage_order = app.storage.user["stage_order"]
        stage_id = stage_order[stage_idx]
        stage = experiment.all_stages[stage_id]

        logger.info("=" * 30)
        logger.info(f"Began stage '{stage.name}' (ID: {stage_id}, Index: {stage_idx})")
        
        # Run stage
        await run_stage(stage, stage_container, button_container)
        logger.info(f"Finished stage '{stage.name}'")
        ui.notify("Loading next task...", type="info", position="top")

        # Save data before moving to next stage
        if isinstance(stage, stages.EnvStage):
            await stage.finish_saving_user_data()
            logger.info(f"Saved data for stage '{stage.name}'")

        # Update stage index
        async with get_user_lock():
            app.storage.user["stage_idx"] = stage_idx + 1

    await finish_experiment(meta_container, stage_container, button_container)

async def finish_experiment(meta_container, stage_container, button_container):
    """Handle experiment completion"""
    nicewebrl.clear_element(meta_container)
    nicewebrl.clear_element(stage_container)
    nicewebrl.clear_element(button_container)
    logger.info("Finishing experiment")
    
    experiment_finished = app.storage.user.get("experiment_finished", False)
    if experiment_finished and not DEBUG:
        return

    if not app.storage.user.get("age"):
        await collect_demographic_info(meta_container)

    # Collect final feedback
    async def submit(feedback):
        app.storage.user["experiment_finished"] = True
        nicewebrl.clear_element(meta_container)
        with meta_container:
            ui.markdown("## Saving data. Please wait")
            ui.markdown(
                "**Once the data is uploaded, this app will automatically move to the next screen**"
            )
        await save_data(final_save=True, feedback=feedback)
        app.storage.user["data_saved"] = True

    app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
    if not app.storage.user["data_saved"]:
        nicewebrl.clear_element(meta_container)
        with meta_container:
            ui.markdown("## Final Feedback")
            ui.markdown(
                "Please provide feedback on the experiment here. For example, please describe if anything went wrong or if you have any suggestions for the experiment."
            )
            text = ui.textarea().style("width: 80%;")
            button = ui.button("Submit")
            async def on_submit_click():
                await submit(text.value)
            button.on('click', on_submit_click)
            while not app.storage.user.get("data_saved", False):
                await asyncio.sleep(0.1)

    # Final screen
    nicewebrl.clear_element(meta_container)
    with meta_container:
        ui.markdown("# Experiment Complete")
        ui.markdown("## Data saved successfully")
        ui.markdown(
            "### Please record the following completion code for compensation:"
        )
        ui.markdown("### **socialrl.cook**")
        ui.markdown("#### You may now close the browser")

async def run_stage(stage, stage_container, button_container):
    """Run a single experiment stage"""
    stage_over_event = asyncio.Event()

    async def local_handle_key_press():
        async with get_user_lock():
            if stage.get_user_data("finished", False):
                logger.info(f"Finished {stage_name(stage)} via key press")
                stage_over_event.set()

    async def handle_button_press():
        if stage.get_user_data("finished", False):
            return
            
        await stage.handle_button_press(stage_container)
        async with get_user_lock():
            if stage.get_user_data("finished", False):
                logger.info(f"Finished {stage_name(stage)} via button press")
                stage_over_event.set()

    # Activate stage
    with stage_container.style("align-items: center;"):
        await stage.activate(stage_container)

    if stage.get_user_data("finished", False):
        logger.info(f"Finished {stage_name(stage)} immediately after activation")
        stage_over_event.set()

    await stage.set_user_data(local_handle_key_press=local_handle_key_press)

    # Setup next button
    with button_container.style("align-items: center;"):
        nicewebrl.clear_element(button_container)
        next_button_container = ui.row()

        async def create_button_and_wait():
            with next_button_container:
                nicewebrl.clear_element(next_button_container)
                button = ui.button("Next page").bind_visibility_from(stage, "next_button")
                await button.clicked()
                logger.info("Button or key pressed")
                await handle_button_press()

        if stage.next_button:
            await create_button_and_wait()

    await stage_over_event.wait()
    nicewebrl.clear_element(button_container)

# ==============================================================================
#                               USER INITIALIZATION
# ==============================================================================
def initialize_user(request: Request):
    """Initialize user session and randomize stage order"""
    logger.info("INITIALIZING USER")
    nicewebrl.initialize_user(seed=DEBUG_SEED)
    
    # Store URL parameters for external platforms (e.g., MTurk)
    app.storage.user["worker_id"] = request.query_params.get("workerId", None)
    app.storage.user["hit_id"] = request.query_params.get("hitId", None)
    app.storage.user["assignment_id"] = request.query_params.get("assignmentId", None)

    app.storage.user["user_id"] = (
        app.storage.user["worker_id"] or app.storage.user["seed"]
    )
    
    def randomize_stage_order():
        """Randomize the order of the prediction tasks, keeping the tutorial first."""
        seed = jax.random.PRNGKey(app.storage.user["seed"])
        # The first stage is the tutorial (index 0), so we shuffle indices 1 to N.
        prediction_task_indices = jax.numpy.arange(1, len(experiment.all_stages))
        shuffled_indices = jax.random.permutation(seed, prediction_task_indices)
        # Combine tutorial stage with shuffled tasks
        stage_order = [0] + [int(i) for i in shuffled_indices]
        return stage_order
    
    app.storage.user["stage_order"] = app.storage.user.get("stage_order", randomize_stage_order())
    app.storage.user["stage_idx"] = app.storage.user.get("stage_idx", 0)
    logger.info("USER INITIALIZED")

# ==============================================================================
#                               UTILITY FUNCTIONS
# ==============================================================================
async def check_if_over(*args, episode_limit=60, **kwargs):
    """Check if experiment should timeout"""
    minutes_passed = nicewebrl.get_user_session_minutes()
    if "session_duration" in app.storage.user:
        minutes_passed = app.storage.user["session_duration"]
    if minutes_passed > episode_limit:
        logger.info(f"experiment timed out after {minutes_passed} minutes")
        app.storage.user["stage_idx"] = len(experiment.all_stages)
        await finish_experiment(*args, **kwargs)

def footer(footer_container):
    """Footer now only contains text labels."""
    with footer_container.classes('w-full justify-end text-right q-pa-sm').style('gap: 2em;'):
        ui.label().bind_text_from(
            app.storage.user, "seed", lambda v: f"User ID: {v}"
        )
        ui.label().bind_text_from(
            app.storage.user,
            "stage_idx",
            lambda v: f"Task: {max(0, int(v))}/{len(experiment.all_stages)-1}" if v > 0 else "Tutorial"
        )

# ==============================================================================
#                               MAIN PAGE ROUTE
# ==============================================================================
@ui.page("/")
async def index(request: Request):
    """Main experiment page"""
    logger.info("INDEX PAGE")
    initialize_user(request)
    ui.run_javascript(f"window.debug = {DEBUG}")

    # Load basic JavaScript
    basic_javascript_file = nicewebrl.basic_javascript_file()
    with open(basic_javascript_file) as f:
        ui.add_body_html("<script>" + f.read() + "</script>")

    # Create main layout
    with ui.header(elevated=True).classes('bg-primary text-white justify-between'):
        ui.label('Agent Behavior Prediction Study').classes('text-h6')
        footer_container_header = ui.row().classes('items-center')

    card = (
        ui.card()
        .classes("w-full max-w-5xl mx-auto my-8")
    )
    
    with card:
        # Add a reactive progress bar at the top of the main card.
        ui.linear_progress().bind_value_from(
            app.storage.user, 'stage_idx',
            lambda idx: max(0, idx) / (len(experiment.all_stages) - 1) if idx is not None else 0.0
        ).props('color=primary')

        episode_limit = 50
        meta_container = ui.column().classes('w-full items-center')
        stage_container = ui.column().classes('w-full')
        button_container = ui.column().classes('w-full')
        ui.timer(
            interval=1,
            callback=lambda: check_if_over(
                episode_limit=episode_limit,
                meta_container=meta_container,
                stage_container=stage_container,
                button_container=button_container,
            ),
        )
        logger.info("STARTING EXPERIMENT")
        await start_experiment(meta_container, stage_container, button_container)
    
    with footer_container_header:
        footer(footer_container_header)

# ==============================================================================
#                               APP STARTUP
# ==============================================================================
ui.run(
    storage_secret="private key to secure the browser session cookie",
    reload="FLY_ALLOC_ID" not in os.environ,
    title="Behavior Prediction",
    show_welcome_message=True,
    on_air=True,
    show=False
)