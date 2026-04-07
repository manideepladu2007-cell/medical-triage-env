import asyncio
import sys
from env.env import MedTriageEnv
from env.models import TriageAction


async def main():
    env = MedTriageEnv(task_id="hard")

    TASK_NAME = "medical-triage"
    ENV_NAME = "medtriage-env"
    MODEL_NAME = "baseline-agent"

    # ---------------- START ---------------- #
    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards = []
    steps_taken = 0

    try:
        obs = await env.reset()

        for step in range(1, 9):

            # ---------------- POLICY ---------------- #
            if obs.vitals == "unknown":
                action = TriageAction(action_type="ask_vitals")

            elif obs.vitals in ["unstable", "slightly_unstable"]:
                action = TriageAction(action_type="send_to_ER")

            else:
                action = TriageAction(action_type="prescribe_basic_meds")

            # ---------------- STEP ---------------- #
            result = await env.step(action)

            reward = result.reward.value
            done = result.done

            rewards.append(reward)
            steps_taken = step

            # ✅ REQUIRED OUTPUT → stdout
            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True
            )

            # 🔥 DEBUG → stderr (SAFE)
            if result.info:
                print(
                    f"DEBUG | reason={result.info.get('reason')} "
                    f"confidence={result.info.get('confidence')}",
                    file=sys.stderr,
                    flush=True
                )

            obs = result.observation

            if done:
                break

        success = sum(rewards) > 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        success = False

    # ---------------- END ---------------- #
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps_taken} rewards={rewards_str}",
        flush=True
    )


if __name__ == "__main__":
    asyncio.run(main())
