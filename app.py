from fastapi import FastAPI, Request
from env.env import MedTriageEnv
from env.models import TriageAction
import asyncio
from inference import main as run_inference

app = FastAPI()

triage_env = MedTriageEnv()


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: Request):

    task_id = "easy"

    try:
        data = await request.json()

        if "task" in data:
            task_id = data["task"]
        elif "task_id" in data:
            task_id = data["task_id"]

    except:
        task_id = request.query_params.get("task", "easy")

    obs = await triage_env.reset(task_id=task_id)

    return obs.model_dump()


@app.post("/step")
async def step(action: TriageAction):

    result = await triage_env.step(action)

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward.value,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def state():
    return triage_env.state()


# RUN INFERENCE FOR LOGS
@app.on_event("startup")
async def start_background():
    asyncio.create_task(run_inference())
