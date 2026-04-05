import asyncio
from env.env import MedTriageEnv
from env.models import TriageAction


async def main():
    # Initialize environment
    env = MedTriageEnv(task_id="hard")

    TASK_NAME = "medical-triage"
    ENV_NAME = "medtriage-env"
    MODEL_NAME = "baseline-agent"

    # START log (MANDATORY FORMAT)
    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}")

    rewards = []
    steps_taken = 0

    try:
        # Reset environment
        obs = await env.reset()

        for step in range(1, 9):  # max 8 steps
            # ---- SIMPLE BASELINE LOGIC ----
            if obs.vitals == "unknown":
                action = TriageAction(action_type="ask_vitals")
            elif obs.vitals in ["unstable", "slightly_unstable"]:
                action = TriageAction(action_type="send_to_ER")
            else:
                action = TriageAction(action_type="prescribe_basic_meds")

            # Take step
            result = await env.step(action)

            reward = result.reward.value
            done = result.done

            rewards.append(reward)
            steps_taken = step

            # STEP log (MANDATORY FORMAT)
            print(
                f"[STEP] step={step} action={action.action_type} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

            obs = result.observation

            if done:
                break

        # Success condition
        success = sum(rewards) > 0

    except Exception as e:
        print(f"[ERROR] {e}")
        success = False

    # END log (MANDATORY FORMAT)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps_taken} rewards={rewards_str}"
    )


if __name__ == "__main__":
    asyncio.run(main())