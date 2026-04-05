from env.env import MedTriageEnv
from env.models import TriageAction
import asyncio


async def run(task):
    env = MedTriageEnv(task_id=task)
    obs = await env.reset()

    actions = [
        TriageAction(action_type="ask_vitals"),
        TriageAction(action_type="send_to_ER"),
    ]

    total = 0
    for a in actions:
        res = await env.step(a)
        total += res.reward.value
        if res.done:
            break

    return total


if __name__ == "__main__":
    print(asyncio.run(run("hard")))