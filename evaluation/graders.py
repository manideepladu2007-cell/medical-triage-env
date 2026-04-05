import asyncio
from env.env import MedTriageEnv
from env.models import TriageAction


# ---------------------------
# SINGLE TASK EVALUATION
# ---------------------------
async def run(task: str):
    env = MedTriageEnv(task_id=task)
    obs = await env.reset()

    total_reward = 0.0
    steps = 0

    # baseline reasoning sequence
    actions = [
        TriageAction(action_type="ask_vitals"),
        TriageAction(action_type="ask_symptom_details"),
        TriageAction(action_type="ask_history"),
        TriageAction(action_type="send_to_ER"),
    ]

    for action in actions:
        result = await env.step(action)

        total_reward += result.reward.value
        steps += 1

        if result.done:
            break

    # ---------------------------
    # NORMALIZATION (0–1 SCORE)
    # ---------------------------
    score = total_reward / 1.5
    score = max(0.0, min(1.0, score))

    return {
        "task": task,
        "score": round(score, 3),
        "steps": steps,
        "total_reward": round(total_reward, 3)
    }


# ---------------------------
# RUN ALL TASKS
# ---------------------------
async def evaluate_all():
    tasks = ["easy", "medium", "hard"]

    results = []
    for t in tasks:
        res = await run(t)
        results.append(res)

    return results


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    results = asyncio.run(evaluate_all())

    print("\n=== EVALUATION RESULTS ===")
    for r in results:
        print(
            f"Task: {r['task']} | Score: {r['score']} | Steps: {r['steps']} | Reward: {r['total_reward']}"
        )
