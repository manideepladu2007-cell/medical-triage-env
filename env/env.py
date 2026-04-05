import random
from env.models import TriageObservation, TriageAction, TriageReward, TASK_REGISTRY


# ---------------------------
# Custom StepResult
# ---------------------------
class StepResult:
    def __init__(self, observation, reward, done, info):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


# ---------------------------
# Environment
# ---------------------------
class MedTriageEnv:

    def __init__(self, task_id="easy", seed=42):
        self.task_id = task_id
        self.rng = random.Random(seed)

    async def reset(self):
        scenario = self.rng.choice(TASK_REGISTRY[self.task_id]["scenarios"])
        self.hidden = scenario["hidden_truth"]

        # slight random variation for realism
        age_variation = self.rng.randint(-3, 3)

        self.state = TriageObservation(
            symptoms=scenario["initial_symptoms"],
            age=scenario["age"] + age_variation,
            known_conditions=[],
            vitals="unknown",
            patient_response=None,
            time_elapsed=0,
            available_actions=self.actions(),
        )

        self.done = False
        self.steps = 0
        self.asked = []

        return self.state

    async def step(self, action: TriageAction):

        info = {}

        if self.done:
            return StepResult(self.state, TriageReward(value=0.0), True, info)

        self.steps += 1
        reward = 0.0

        # ---------------------------
        # QUESTION ACTIONS
        # ---------------------------
        if action.action_type in ["ask_symptom_details", "ask_vitals", "ask_history"]:

            if action.action_type in self.asked:
                reward = -0.05
                info["reason"] = "Repeated question"

            else:
                self.asked.append(action.action_type)

                if action.action_type in self.hidden["useful_questions"]:
                    reward = 0.1
                    info["reason"] = "Useful question"
                else:
                    reward = 0.03
                    info["reason"] = "Less useful question"

            response = self.hidden["question_responses"].get(
                action.action_type, "Patient unsure"
            )
            self.state.patient_response = response

            if action.action_type == "ask_vitals":
                self.state.vitals = self.hidden["revealed_vitals"]

        # ---------------------------
        # FINAL DECISION ACTIONS
        # ---------------------------
        else:
            self.done = True

            if action.action_type == self.hidden["correct_action"]:
                reward = 1.0
                info["reason"] = "Correct decision"

            elif self.hidden["severity"] == "critical":
                reward = -1.0
                info["reason"] = "Missed critical condition"

            else:
                reward = -0.4
                info["reason"] = "Incorrect decision"

            # penalty for blind decision
            if len(self.asked) == 0:
                reward -= 0.2
                info["penalty"] = "No prior questioning"

        # ---------------------------
        # BONUS FOR GOOD EXPLORATION
        # ---------------------------
        if not self.done and len(self.asked) >= 2:
            reward += 0.05
            info["bonus"] = "Good exploration"

        # ---------------------------
        # TIME PENALTY (HARD TASK)
        # ---------------------------
        if self.task_id == "hard" and not self.done and self.steps > 6:
            penalty = 0.05 * (self.steps - 6)
            reward -= penalty
            info["time_penalty"] = round(penalty, 2)

        # ---------------------------
        # CONFIDENCE SCORE (NEW FEATURE)
        # ---------------------------
        confidence = 0.5

        if self.state.vitals == "unstable":
            confidence += 0.3

        if len(self.asked) >= 2:
            confidence += 0.2

        confidence = min(1.0, confidence)
        info["confidence"] = round(confidence, 2)

        # ---------------------------
        # UPDATE STATE
        # ---------------------------
        self.state.time_elapsed = self.steps

        return StepResult(
            observation=self.state,
            reward=TriageReward(value=round(reward, 3), breakdown=info),
            done=self.done,
            info=info
        )

    def state(self):
        return self.state

    def actions(self):
        return [
            "ask_symptom_details",
            "ask_vitals",
            "ask_history",
            "send_to_ER",
            "schedule_doctor",
            "prescribe_basic_meds",
            "ignore_case",
        ]
