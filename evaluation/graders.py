from env.env import MedTriageEnv
from env.models import TriageAction


async def run(task_id: str):
    env = MedTriageEnv(task_id=task_id)
    obs = await env.reset()

    total_reward = 0

    actions = [
        TriageAction(action_type="ask_vitals"),
        TriageAction(action_type="ask_symptom_details"),
        TriageAction(action_type="send_to_ER"),
    ]

    for action in actions:
        result = await env.step(action)
        total_reward += result.reward.value

        if result.done:
            break

    score = max(0.0, min(1.0, total_reward / 1.5))
    return score


async def grade_easy():
    return await run("easy")


async def grade_medium():
    return await run("medium")


async def grade_hard():
    return await run("hard")
