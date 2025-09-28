import asyncio
import os
import random
import struct

import aiofiles
import msgpack
from fastapi import Request
from nicegui import app, ui

# This now imports from your new experiment file.
import human_predict_ai_exp as experiment
import nicewebrl
from nicewebrl import stages
from nicewebrl.logging import get_logger, setup_logging

# Add static files path
app.add_static_files('/human_vids_curated', 'human_vids_curated')

# ==============================================================================
#                               CONFIGURATION
# ==============================================================================
DATA_DIR = os.environ.get("DATA_DIR", "data")
DEBUG = int(os.environ.get("DEBUG", 0))
DEBUG_SEED = int(os.environ.get("SEED", 0))
NAME = os.environ.get("NAME", "exp_method1")

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
def get_user_lock():
    user_seed = app.storage.user["seed"]
    if user_seed not in _user_locks:
        _user_locks[user_seed] = asyncio.Lock()
    return _user_locks[user_seed]

async def experiment_not_finished():
    async with get_user_lock():
        stage_idx = app.storage.user.get("stage_idx", 0)
        not_finished = not app.storage.user.get("experiment_finished", False)
        not_finished &= stage_idx < len(experiment.all_stages)
    return not_finished

async def save_data(final_save=True, feedback=None, **kwargs):
    """Save user data to file"""
    user_data_file = experiment.get_user_save_file_fn()
    if final_save:
        user_storage = nicewebrl.make_serializable(dict(app.storage.user))
        last_line = dict(finished=True, feedback=feedback, user_storage=user_storage, **kwargs)
        async with aiofiles.open(user_data_file, "ab") as f:
            packed_data = msgpack.packb(last_line)
            await f.write(struct.pack(">i", len(packed_data)))
            await f.write(packed_data)

# ==============================================================================
#                       CONSENT FORM AND DEMOGRAPHICS
# ==============================================================================
async def make_consent_form(container):
    """Display consent form and wait for agreement"""
    consent_given = asyncio.Event()
    with container:
        ui.markdown("## Consent Form")
        # Ensure you have a 'consent.md' file in a 'misc' directory
        try:
            with open("misc/consent.md", "r") as consent_file:
                consent_text = consent_file.read()
            ui.markdown(consent_text)
        except FileNotFoundError:
            ui.markdown("*(Consent form file not found. Please create 'misc/consent.md')*")
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
            if age_input.value is None:
                ui.notify("Please enter a valid age.", type="warning")
                return
            app.storage.user["age"] = int(age_input.value)
            app.storage.user["sex"] = sex_input.value
            success = True
        button = ui.button("Submit", on_click=submit)
        while not success:
            await asyncio.sleep(0.1)

# ==============================================================================
#                               EXPERIMENT FLOW
# ==============================================================================
async def start_experiment(meta_container, stage_container):
    logger.info("STARTING EXPERIMENT (METHOD 1)")
    
    if not (app.storage.user.get("experiment_started", False) or DEBUG):
        await make_consent_form(stage_container)
        app.storage.user["experiment_started"] = True

    while await experiment_not_finished():
        nicewebrl.clear_element(stage_container)
        stage_idx = app.storage.user["stage_idx"]
        stage_order = app.storage.user["stage_order"]
        stage_id = stage_order[stage_idx]
        stage = experiment.all_stages[stage_id]
        logger.info("=" * 30)
        logger.info(f"Began stage '{stage.name or f'Task {stage_idx}'}' (ID: {stage_id}, Index: {stage_idx})")
        
        await stage.display_fn(stage, stage_container)
        
        logger.info(f"Finished stage '{stage.name or f'Task {stage_idx}'}'")
        if stage_idx < len(experiment.all_stages) - 1:
            ui.notify("Loading next task...", type="info", position="top")

        async with get_user_lock():
            app.storage.user["stage_idx"] = stage_idx + 1

    await finish_experiment(meta_container, stage_container)

async def finish_experiment(meta_container, stage_container):
    nicewebrl.clear_element(meta_container)
    nicewebrl.clear_element(stage_container)
    logger.info("Finishing experiment")
    
    experiment_finished = app.storage.user.get("experiment_finished", False)
    if experiment_finished and not DEBUG:
        return

    if not app.storage.user.get("age"):
        await collect_demographic_info(meta_container)

    async def submit_feedback(feedback):
        app.storage.user["experiment_finished"] = True
        nicewebrl.clear_element(meta_container)
        with meta_container:
            ui.markdown("## Saving data. Please wait...")
        await save_data(final_save=True, feedback=feedback)
        app.storage.user["data_saved"] = True

    app.storage.user["data_saved"] = app.storage.user.get("data_saved", False)
    if not app.storage.user["data_saved"]:
        nicewebrl.clear_element(meta_container)
        with meta_container:
            ui.markdown("## Final Feedback")
            ui.markdown("Please provide feedback on the experiment. For example, please describe if anything went wrong or if you have any suggestions.")
            text = ui.textarea().style("width: 80%;")
            button = ui.button("Submit", on_click=lambda: submit_feedback(text.value))
            while not app.storage.user.get("data_saved", False):
                await asyncio.sleep(0.1)

    nicewebrl.clear_element(meta_container)
    with meta_container:
        ui.markdown("# Experiment Complete")
        ui.markdown("## Thank you for your participation!")
        ui.markdown("### Please record the following completion code for compensation:")
        ui.markdown("### **socialrl.cook**")
        ui.markdown("#### You may now close the browser.")

# ==============================================================================
#                               USER INITIALIZATION
# ==============================================================================
def initialize_user(request: Request):
    if "stage_idx" in app.storage.user:
        return
    logger.info("INITIALIZING USER")
    nicewebrl.initialize_user(seed=DEBUG_SEED)
    app.storage.user["user_id"] = request.query_params.get("workerId") or app.storage.user["seed"]
    prediction_task_indices = list(range(1, len(experiment.all_stages)))
    random.seed(app.storage.user["seed"])
    random.shuffle(prediction_task_indices)
    stage_order = [0] + prediction_task_indices
    app.storage.user["stage_order"] = stage_order
    app.storage.user["stage_idx"] = 0
    logger.info("USER INITIALIZED")

# ==============================================================================
#                                MAIN PAGE ROUTE
# ==============================================================================
@ui.page("/")
async def index(request: Request):
    initialize_user(request)
    with ui.header(elevated=True).classes('bg-primary text-white justify-between'):
        ui.label('Agent Behavior Prediction Study').classes('text-h6')
        ui.label().bind_text_from(
            app.storage.user, "stage_idx",
            lambda v: f"Task: {max(0, int(v))}/{len(experiment.all_stages)-1}" if v > 0 else "Tutorial"
        )
    with ui.card().classes("w-full max-w-5xl mx-auto my-8"):
        meta_container = ui.column().classes('w-full items-center')
        stage_container = ui.column().classes('w-full')
        await start_experiment(meta_container, stage_container)

# ==============================================================================
#                                 APP STARTUP
# ==============================================================================
ui.run(
    storage_secret="a_very_secret_key_for_this_experiment",
    reload="FLY_ALLOC_ID" not in os.environ,
    title="Behavior Prediction (Method 1)",
    on_air=True,
    show=False
)