import asyncio
import sys
from env.env import MedTriageEnv
from env.models import TriageAction


async def main():
    env = MedTriageEnv(task_id="hard")

    TASK_NAME = "medical-triage"
    ENV_NAME = "medtriage-env"
    MODEL_NAME = "baseline-agent"

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards = []
    steps_taken = 0
    debug_logs = []  # ✅ collect debug info

    try:
        obs = await env.reset()

        for step in range(1, 9):

            # POLICY
            if obs.vitals == "unknown":
                action = TriageAction(action_type="ask_vitals")
            elif obs.vitals in ["unstable", "slightly_unstable"]:
                action = TriageAction(action_type="send_to_ER")
            else:
                action = TriageAction(action_type="prescribe_basic_meds")

            result = await env.step(action)

            reward = result.reward.value
            done = result.done

            rewards.append(reward)
            steps_taken = step

            # ✅ STRICT OUTPUT
            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            # 🔥 STORE DEBUG (DON'T PRINT YET)
            if result.info:
                debug_logs.append(
                    f"DEBUG | step={step} reason={result.info.get('reason')} "
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

    # 🔥 PRINT DEBUG AFTER END (TO STDERR)
    for log in debug_logs:
        print(log, file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
