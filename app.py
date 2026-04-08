from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

from env.env import MedTriageEnv
from env.models import TriageAction
from inference import main as run_inference  # 👈 ADD THIS

app = FastAPI()

env = MedTriageEnv(task_id="hard")


class StepRequest(BaseModel):
    action_type: str


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/reset")
async def reset():
    obs = await env.reset()
    return obs.model_dump()


@app.post("/step")
async def step(req: StepRequest):
    action = TriageAction(action_type=req.action_type)
    result = await env.step(action)

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward.value,
        "done": result.done,
        "info": result.info
    }


# 🚀 THIS BRINGS BACK YOUR LOGS
@app.on_event("startup")
async def start_background_task():
    asyncio.create_task(run_inference())
