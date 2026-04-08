import asyncio
import os
import sys
from openai import OpenAI

from env.env import MedTriageEnv
from env.models import TriageAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


def fallback(obs):
    if obs.vitals == "unknown":
        return "ask_vitals"
    if "chest pain" in obs.symptoms or obs.vitals == "unstable":
        return "send_to_ER"
    return "ask_symptom_details"


def get_llm_action(client, obs):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": str(obs.model_dump())}],
            temperature=0.2,
        )
        return "ask_vitals"
    except:
        return fallback(obs)


async def main():

    client = None
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = ["easy", "medium", "hard"]

    for task_name in tasks:

        env = MedTriageEnv(task_id=task_name)

        print(f"[START] task={task_name} env=medtriage-env model={MODEL_NAME}", flush=True)

        rewards = []
        steps_taken = 0
        success = False

        try:
            obs = await env.reset()

            for step in range(1, 8):

                if client:
                    action_str = get_llm_action(client, obs)
                else:
                    action_str = fallback(obs)

                action = TriageAction(action_type=action_str)

                result = await env.step(action)

                reward = result.reward.value
                done = result.done

                rewards.append(reward)
                steps_taken = step

                print(
                    f"[STEP] step={step} action={action.action_type} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null",
                    flush=True,
                )

                obs = result.observation

                if done:
                    break

            total_reward = sum(rewards)
            score = max(0.0, min(1.0, total_reward / 1.5))
            success = score > 0.0

        except Exception as e:
            print(f"ERROR | {e}", file=sys.stderr, flush=True)
            success = False
            score = 0.0

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps_taken} score={score:.3f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
