from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------
# OBSERVATION
# ---------------------------
class TriageObservation(BaseModel):
    symptoms: list[str]
    age: int
    known_conditions: list[str] = Field(default_factory=list)
    vitals: str = "unknown"
    patient_response: Optional[str] = None
    time_elapsed: int = 0
    available_actions: list[str] = Field(default_factory=list)


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
    breakdown: dict = Field(default_factory=dict)


# ---------------------------
# TASK REGISTRY
# ---------------------------
TASK_REGISTRY = {

    # ---------------- EASY ---------------- #
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

    # ---------------- MEDIUM ---------------- #
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
            },
            {
                "initial_symptoms": ["headache", "blurred vision"],
                "age": 35,
                "hidden_truth": {
                    "correct_action": "schedule_doctor",
                    "severity": "medium",
                    "useful_questions": ["ask_history"],
                    "question_responses": {
                        "ask_symptom_details": "Persistent headache",
                        "ask_history": "Migraine history",
                    },
                    "revealed_vitals": "stable",
                },
            },
            {
                # 🔥 Added realism case
                "initial_symptoms": ["fatigue", "weight loss"],
                "age": 50,
                "hidden_truth": {
                    "correct_action": "schedule_doctor",
                    "severity": "medium",
                    "useful_questions": ["ask_history"],
                    "question_responses": {
                        "ask_symptom_details": "Gradual fatigue",
                        "ask_history": "Diabetes",
                    },
                    "revealed_vitals": "stable",
                },
            }
        ],
    },

    # ---------------- HARD ---------------- #
    "hard": {
        "scenarios": [
            # Classic emergency
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
            },

            # 🔥 Stroke disguised as dizziness
            {
                "initial_symptoms": ["dizziness", "nausea"],
                "age": 60,
                "hidden_truth": {
                    "correct_action": "send_to_ER",
                    "severity": "critical",
                    "useful_questions": ["ask_vitals", "ask_history"],
                    "question_responses": {
                        "ask_symptom_details": "Sudden dizziness and imbalance",
                        "ask_vitals": "Very unstable",
                        "ask_history": "Hypertension",
                    },
                    "revealed_vitals": "unstable",
                },
            },

            # 🔥 Heart attack disguised as anxiety
            {
                "initial_symptoms": ["anxiety", "chest discomfort"],
                "age": 52,
                "hidden_truth": {
                    "correct_action": "send_to_ER",
                    "severity": "critical",
                    "useful_questions": ["ask_vitals"],
                    "question_responses": {
                        "ask_symptom_details": "Feels like panic attack",
                        "ask_vitals": "Unstable",
                    },
                    "revealed_vitals": "unstable",
                },
            }
        ],
    },
}
