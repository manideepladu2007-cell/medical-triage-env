import random
from env.models import TriageObservation, TriageAction, TriageReward, TASK_REGISTRY


# ---------------------------
# Custom StepResult (NO openenv dependency)
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

        self.state = TriageObservation(
            symptoms=scenario["initial_symptoms"],
            age=scenario["age"],
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
        if self.done:
            return StepResult(self.state, TriageReward(value=0.0), True, {})

        self.steps += 1
        reward = 0.0

        # ---------------------------
        # QUESTION ACTIONS
        # ---------------------------
        if action.action_type in ["ask_symptom_details", "ask_vitals", "ask_history"]:

            if action.action_type in self.asked:
                reward = -0.05
            else:
                self.asked.append(action.action_type)
                if action.action_type in self.hidden["useful_questions"]:
                    reward = 0.1
                else:
                    reward = 0.03

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

            elif self.hidden["severity"] == "critical":
                reward = -1.0

            else:
                reward = -0.4

        # ---------------------------
        # TIME PENALTY (HARD TASK)
        # ---------------------------
        if self.task_id == "hard" and not self.done and self.steps > 6:
            reward -= 0.05 * (self.steps - 6)

        self.state.time_elapsed = self.steps

        return StepResult(
            observation=self.state,
            reward=TriageReward(value=round(reward, 3)),
            done=self.done,
            info={}
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