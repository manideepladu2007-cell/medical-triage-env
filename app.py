from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

from env.env import MedTriageEnv
from env.models import TriageAction

app = FastAPI()

env = MedTriageEnv(task_id="hard")


# ---------------------------
# REQUEST MODELS
# ---------------------------
class StepRequest(BaseModel):
    action_type: str


# ---------------------------
# ROOT (for HF health check)
# ---------------------------
@app.get("/")
def root():
    return {"status": "running"}


# ---------------------------
# RESET ENDPOINT ✅
# ---------------------------
@app.post("/reset")
async def reset():
    obs = await env.reset()
    return obs.model_dump()


# ---------------------------
# STEP ENDPOINT ✅
# ---------------------------
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
