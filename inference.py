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
# FALLBACK LOGIC
# ---------------------------
def fallback_decision(obs):
    if obs.vitals == "unknown":
        return "ask_vitals", "Fallback: need vitals"
    elif obs.vitals in ["unstable", "slightly_unstable"]:
        return "send_to_ER", "Fallback: unstable condition"
    else:
        return "prescribe_basic_meds", "Fallback: mild condition"


# ---------------------------
# LLM CALL
# ---------------------------
def get_llm_decision(client, obs):
    prompt = f"""
You are a medical triage agent.

Patient:
{obs.model_dump_json()}

Choose ONE action from:
["ask_symptom_details","ask_vitals","ask_history","send_to_ER","schedule_doctor","prescribe_basic_meds"]

Respond ONLY in JSON:
{{"action_type": "...", "reasoning": "..."}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()

        # JSON cleanup fix
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()

        data = json.loads(content)

        action = data.get("action_type", "ask_vitals")
        reason = data.get("reasoning", "Fallback reasoning")

        return action, reason

    except Exception as e:
        print(f"DEBUG | LLM failed: {e}", file=sys.stderr, flush=True)
        return fallback_decision(obs)


# ---------------------------
# MAIN
# ---------------------------
async def main():

    # ✅ SAFE CLIENT INITIALIZATION
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        client = None

    env = MedTriageEnv(task_id="hard")

    print(f"[START] task=medical-triage env=medtriage-env model={MODEL_NAME}", flush=True)

    rewards = []
    steps_taken = 0
    debug_logs = []

    try:
        obs = await env.reset()

        for step in range(1, 9):

            # ✅ SAFE DECISION LOGIC
            if client:
                action_str, reasoning = get_llm_decision(client, obs)
            else:
                action_str, reasoning = fallback_decision(obs)

            action = TriageAction(
                action_type=action_str,
                reasoning=reasoning
            )

            result = await env.step(action)

            reward = result.reward.value
            done = result.done

            rewards.append(reward)
            steps_taken = step

            # ✅ STRICT OUTPUT FORMAT
            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            # DEBUG LOG (stderr safe)
            debug_logs.append(
                f"DEBUG | step={step} llm_reason={reasoning} "
                f"env_reason={result.info.get('reason')} "
                f"confidence={result.info.get('confidence')}"
            )

            obs = result.observation

            if done:
                break

        # ✅ SCORE CALCULATION
        total_reward = sum(rewards)
        score = total_reward / 1.5
        score = max(0.0, min(1.0, score))
        success = score > 0.0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        success = False
        score = 0.0

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # ✅ FINAL OUTPUT (WITH SCORE)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps_taken} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

    # DEBUG AFTER END
    for log in debug_logs:
        print(log, file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
