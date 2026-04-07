import asyncio
import sys
import json
from env.env import MedTriageEnv
from env.models import TriageAction


# ---------------------------
# MOCK LLM AGENT
# ---------------------------
def mock_llm_decision(observation):
    """
    Simulates LLM reasoning over observation.
    """

    obs_dict = observation.model_dump()

    # Simulated "reasoning"
    if obs_dict["vitals"] == "unknown":
        action = "ask_vitals"
        reasoning = "Vitals unknown, need more information"

    elif obs_dict["vitals"] in ["unstable", "slightly_unstable"]:
        action = "send_to_ER"
        reasoning = "Vitals indicate risk, escalate immediately"

    else:
        action = "prescribe_basic_meds"
        reasoning = "Condition appears mild"

    return action, reasoning


# ---------------------------
# MAIN
# ---------------------------
async def main():
    env = MedTriageEnv(task_id="hard")

    TASK_NAME = "medical-triage"
    ENV_NAME = "medtriage-env"
    MODEL_NAME = "mock-llm-agent"  # 🔥 important change

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards = []
    steps_taken = 0
    debug_logs = []

    try:
        obs = await env.reset()

        for step in range(1, 9):

            # 🔥 LLM-style decision
            action_str, reasoning = mock_llm_decision(obs)

            action = TriageAction(
                action_type=action_str,
                reasoning=reasoning
            )

            result = await env.step(action)

            reward = result.reward.value
            done = result.done

            rewards.append(reward)
            steps_taken = step

            # REQUIRED OUTPUT
            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            # STORE DEBUG
            debug_logs.append(
                f"DEBUG | step={step} llm_reason='{reasoning}' "
                f"env_reason='{result.info.get('reason')}' "
                f"confidence={result.info.get('confidence')}"
            )

            obs = result.observation

            if done:
                break

        success = sum(rewards) > 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps_taken} rewards={rewards_str}",
        flush=True
    )

    # 🔥 DEBUG AFTER END (SAFE)
    for log in debug_logs:
        print(log, file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
