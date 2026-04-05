from pydantic import BaseModel
from typing import Optional


# ---------------------------
# OBSERVATION
# ---------------------------
class TriageObservation(BaseModel):
    symptoms: list[str]
    age: int
    known_conditions: list[str] = []
    vitals: str = "unknown"
    patient_response: Optional[str] = None
    time_elapsed: int = 0
    available_actions: list[str] = []


# ---------------------------
# ACTION
# ---------------------------
class TriageAction(BaseModel):
    action_type: str
    reasoning: Optional[str] = None


# ---------------------------
# REWARD
# ---------------------------
class TriageReward(BaseModel):
    value: float
    breakdown: dict = {}


# ---------------------------
# TASK REGISTRY
# ---------------------------
TASK_REGISTRY = {
    "easy": {
        "scenarios": [
            {
                "initial_symptoms": ["fever", "cough"],
                "age": 25,
                "hidden_truth": {
                    "correct_action": "prescribe_basic_meds",
                    "severity": "low",
                    "useful_questions": ["ask_symptom_details"],
                    "question_responses": {
                        "ask_symptom_details": "Mild fever",
                        "ask_vitals": "Stable",
                        "ask_history": "No issues",
                    },
                    "revealed_vitals": "stable",
                },
            }
        ],
    },

    "medium": {
        "scenarios": [
            {
                "initial_symptoms": ["chest tightness"],
                "age": 45,
                "hidden_truth": {
                    "correct_action": "schedule_doctor",
                    "severity": "medium",
                    "useful_questions": ["ask_vitals"],
                    "question_responses": {
                        "ask_symptom_details": "Gradual onset",
                        "ask_vitals": "Slightly unstable",
                    },
                    "revealed_vitals": "slightly_unstable",
                },
            }
        ],
    },

    "hard": {
        "scenarios": [
            {
                "initial_symptoms": ["chest pain", "sweating"],
                "age": 55,
                "hidden_truth": {
                    "correct_action": "send_to_ER",
                    "severity": "critical",
                    "useful_questions": ["ask_vitals"],
                    "question_responses": {
                        "ask_symptom_details": "Severe pain",
                        "ask_vitals": "Unstable",
                    },
                    "revealed_vitals": "unstable",
                },
            }
        ],
    },
}