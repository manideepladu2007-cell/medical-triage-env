import random
from typing import Dict, Any

from env.models import (
    TriageObservation,
    TriageAction,
    TriageReward,
    TASK_REGISTRY,
)


class StepResult:
    def __init__(self, observation, reward, done, info):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class MedTriageEnv:
    def __init__(self, task_id="easy", seed=42):
        self.task_id = task_id
        self.rng = random.Random(seed)

    async def reset(self):
        self.current_case = self.rng.choice(
            TASK_REGISTRY[self.task_id]["scenarios"]
        )
        self.step_count = 0
        self.done = False
        self.asked_questions = set()

        self.current_vitals = "unknown"

        return TriageObservation(
            symptoms=self.current_case["initial_symptoms"],
            age=self.current_case["age"],
            known_conditions=[],
            vitals="unknown",
            available_actions=[
                "ask_symptom_details",
                "ask_vitals",
                "ask_history",
                "send_to_ER",
                "schedule_doctor",
                "prescribe_basic_meds",
            ],
        )

    async def step(self, action: TriageAction):
        if self.done:
            return StepResult(None, TriageReward(value=0.0), True, {})

        self.step_count += 1
        hidden = self.current_case["hidden_truth"]

        reward = 0.0
        info = {"reason": "", "confidence": 0.0}

        # ---------------- ASK QUESTIONS ---------------- #
        if action.action_type.startswith("ask"):
            if action.action_type in self.asked_questions:
                reward -= 0.05
                info["reason"] = "Repeated question"
                info["confidence"] = 0.3
            else:
                self.asked_questions.add(action.action_type)

                if action.action_type in hidden["useful_questions"]:
                    reward += 0.1
                    info["reason"] = "Useful question"
                    info["confidence"] = 0.8
                else:
                    reward -= 0.02
                    info["reason"] = "Irrelevant question"
                    info["confidence"] = 0.5

                if action.action_type == "ask_vitals":
                    self.current_vitals = hidden["revealed_vitals"]

            obs = TriageObservation(
                symptoms=self.current_case["initial_symptoms"],
                age=self.current_case["age"],
                known_conditions=[],
                vitals=self.current_vitals,
                patient_response=hidden["question_responses"].get(
                    action.action_type, "No response"
                ),
                time_elapsed=self.step_count,
                available_actions=[
                    "ask_symptom_details",
                    "ask_vitals",
                    "ask_history",
                    "send_to_ER",
                    "schedule_doctor",
                    "prescribe_basic_meds",
                ],
            )

            return StepResult(obs, TriageReward(value=reward), False, info)

        # ---------------- FINAL DECISION ---------------- #
        correct = hidden["correct_action"]

        if action.action_type == correct:
            reward += 1.0
            info["reason"] = "Correct decision"
            info["confidence"] = 0.8
        else:
            if hidden["severity"] == "critical":
                reward -= 1.0
                info["reason"] = "Missed critical condition"
                info["confidence"] = 1.0
            else:
                reward -= 0.2
                info["reason"] = "Suboptimal decision"
                info["confidence"] = 0.6

        self.done = True

        return StepResult(
            TriageObservation(
                symptoms=[],
                age=0,
                available_actions=[],
            ),
            TriageReward(value=reward),
            True,
            info,
        )

    # ✅ REQUIRED FOR OPENENV VALIDATION
    def state(self) -> Dict[str, Any]:
        return {
            "step": getattr(self, "step_count", 0),
            "symptoms": self.current_case["initial_symptoms"]
            if hasattr(self, "current_case")
            else [],
            "age": self.current_case["age"]
            if hasattr(self, "current_case")
            else 0,
        }
