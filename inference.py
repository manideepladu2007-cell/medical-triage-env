import asyncio
import os
import sys
import json

from openai import OpenAI

from env.env import MedTriageEnv
from env.models import TriageAction


# ---------------- ENV VARIABLES ---------------- #
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

HF_TOKEN = os.getenv("HF_TOKEN")  # MUST exist (no default)
API_KEY = HF_TOKEN


# ---------------- SYSTEM PROMPT ---------------- #
SYSTEM_PROMPT = """
You are a medical triage agent.

Choose ONE action from:
["ask_symptom_details","ask_vitals","ask_history",
"send_to_ER","schedule_doctor","prescribe_basic_meds"]

Respond ONLY JSON:
{"action_type":"...","reasoning":"..."}
"""


# ---------------- LLM FUNCTION ---------------- #
def get_llm_action(client, obs_json):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_json},
            ],
            temperature=0.2,
            max_tokens=120,
        )

        content = response.choices[0].message.content.strip()

        # 🔧 JSON CLEANING (SAFE)
        if "```" in content:
            content = content.split("```")[-2]

        data = json.loads(content)

        return TriageAction(
            action_type=data.get("action_type", "ask_vitals"),
            reasoning=data.get("reasoning", ""),
        )

    except Exception as e:
        print(f"DEBUG | LLM failed: {e}", file=sys.stderr, flush=True)

        # 🔥 FALLBACK (SAFE)
        return TriageAction(
            action_type="ask_vitals",
            reasoning="Fallback: need vitals",
        )


# ---------------- MAIN ---------------- #
async def main():
    env = MedTriageEnv(task_id="hard")

    TASK_NAME = "medical-triage"
    ENV_NAME = "medtriage-env"

    # Init client ONLY if token exists
    client = None
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(
        f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}",
        flush=True,
    )

    rewards = []
    steps_taken = 0
    success = False

    try:
        obs = await env.reset()

        for step in range(1, 8):
            # ---------------- ACTION ---------------- #
            if client:
                action = get_llm_action(client, obs.model_dump_json())
            else:
                action = TriageAction(
                    action_type="ask_vitals",
                    reasoning="No API key fallback",
                )

            # ---------------- STEP ---------------- #
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

            # DEBUG → stderr
            if result.info:
                print(
                    f"DEBUG | step={step} llm_reason={action.reasoning} "
                    f"env_reason={result.info.get('reason')} "
                    f"confidence={result.info.get('confidence')}",
                    file=sys.stderr,
                    flush=True,
                )

            obs = result.observation

            if done:
                break

        # ---------------- SCORE ---------------- #
        total_reward = sum(rewards)
        score = total_reward / 1.5
        score = max(0.0, min(1.0, score))
        success = score > 0.0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        success = False
        score = 0.0

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ---------------- END ---------------- #
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps_taken} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
