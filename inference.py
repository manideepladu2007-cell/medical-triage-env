import asyncio
import os
import sys
import json
from openai import OpenAI

from env.env import MedTriageEnv
from env.models import TriageAction


# ---------------------------
# ENV VARIABLES
# ---------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


# ---------------------------
# SAFE LLM CALL
# ---------------------------
def get_llm_action(client, observation):
    """
    Calls LLM safely. If fails → fallback logic.
    """

    # If no API → fallback immediately
    if not API_KEY:
        return fallback_decision(observation, "No API key fallback")

    prompt = f"""
You are a medical triage agent.

Patient:
{observation.model_dump_json()}

Choose ONE action from:
["ask_symptom_details", "ask_vitals", "ask_history",
"send_to_ER", "schedule_doctor", "prescribe_basic_meds"]

Respond ONLY in JSON:
{{"action_type": "...", "reasoning": "..."}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )

        raw = response.choices[0].message.content.strip()

        # 🛡 CLEAN JSON (handles ```json blocks)
        if "```" in raw:
            raw = raw.split("```")[-2].strip()

        data = json.loads(raw)

        action = data.get("action_type", "ask_vitals")
        reasoning = data.get("reasoning", "LLM decision")

        return action, reasoning

    except Exception as e:
        return fallback_decision(observation, f"LLM error: {e}")


# ---------------------------
# FALLBACK POLICY (SAFE)
# ---------------------------
def fallback_decision(obs, reason):
    """
    Simple safe policy when LLM fails.
    """

    if obs.vitals == "unknown":
        return "ask_vitals", reason

    if "chest pain" in obs.symptoms or obs.vitals == "unstable":
        return "send_to_ER", reason

    return "ask_symptom_details", reason


# ---------------------------
# MAIN
# ---------------------------
async def main():

    client = None
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = MedTriageEnv(task_id="hard")

    TASK_NAME = "medical-triage"
    ENV_NAME = "medtriage-env"

    rewards = []
    steps_taken = 0
    success = False

    # ---------------- START ---------------- #
    print(
        f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}",
        flush=True
    )

    try:
        obs = await env.reset()

        for step in range(1, 9):

            # ---- GET ACTION ---- #
            action_str, llm_reason = get_llm_action(client, obs)

            action = TriageAction(action_type=action_str)

            # ---- STEP ---- #
            result = await env.step(action)

            reward = result.reward.value
            done = result.done

            rewards.append(reward)
            steps_taken = step

            # ✅ STRICT STDOUT (ONLY THIS FORMAT)
            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            # 🔥 DEBUG → STDERR ONLY (SAFE)
            if result.info:
                env_reason = result.info.get("reason", "none")

                print(
                    f"DEBUG | step={step} llm_reason={llm_reason} "
                    f"env_reason={env_reason}",
                    file=sys.stderr,
                    flush=True
                )

            obs = result.observation

            if done:
                break

        # ---- SCORE ---- #
        total_reward = sum(rewards)
        score = total_reward / 1.5
        score = max(0.0, min(1.0, score))

        success = score > 0.0

    except Exception as e:
        print(f"ERROR | {e}", file=sys.stderr, flush=True)
        success = False
        score = 0.0

    # ---------------- END ---------------- #
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps_taken} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# ---------------------------
# ENTRY
# ---------------------------
if __name__ == "__main__":
    asyncio.run(main())
