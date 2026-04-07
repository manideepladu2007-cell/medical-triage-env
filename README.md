# 🏥 Clinical Triage Environment (OpenEnv RL)

## 🚀 Overview

This project implements a **real-world medical triage simulation environment** using the OpenEnv framework.
It models how clinicians assess patients under uncertainty, gather information, and make critical decisions.

The environment is designed for **LLM-based agents** to interact, reason, and act through structured observations, actions, and rewards.

---

## 🎯 Key Features

### 🧠 Real-World Simulation

* Simulates clinical triage scenarios (fever, chest pain, dizziness, etc.)
* Models **uncertain and misleading symptoms**
* Includes **hidden conditions** (e.g., stroke disguised as dizziness)

---

### 🔍 Multi-Step Decision Making

Agents must:

1. Ask relevant questions (vitals, history, symptoms)
2. Interpret patient responses
3. Make a final decision (ER / doctor / medication)

---

### ⚠️ Adversarial & Edge Cases

* Critical conditions masked as mild symptoms
* Misleading patient responses
* Forces **reasoning over pattern matching**

---

### 🎯 Meaningful Reward Function

* ✅ +0.10 → Asking useful questions
* ❌ -0.05 → Repeating questions
* ❌ -1.00 → Missing critical condition
* ✅ +1.00 → Correct final decision

Encourages **safe, efficient, and intelligent decision-making**

---

### 🤖 LLM-Compatible Environment

* Fully compatible with OpenAI client
* Uses:

  * `API_BASE_URL`
  * `MODEL_NAME`
  * `HF_TOKEN`
* Includes **robust fallback policy** to ensure stability even if API fails

---

## 🧪 Tasks

| Difficulty | Description                                       |
| ---------- | ------------------------------------------------- |
| Easy       | Simple symptoms (fever, cough)                    |
| Medium     | Ambiguous cases (headache, blurred vision)        |
| Hard       | Critical hidden conditions (heart attack, stroke) |

---

## ⚙️ OpenEnv Compliance

✔ Typed models using Pydantic
✔ `reset()`, `step()`, `state()` implemented
✔ Deterministic grader (0.0 → 1.0 score)
✔ Structured environment interactions

---

## 📊 Example Output

```
[START] task=medical-triage env=medtriage-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=ask_vitals reward=0.10 done=false error=null
[STEP] step=2 action=send_to_ER reward=1.00 done=true error=null
[END] success=true steps=2 score=0.733 rewards=0.10,1.00
```

---

## 🏗️ Architecture

* `env/` → Environment logic & reward system
* `evaluation/` → Graders
* `inference.py` → LLM-driven agent
* `app.py` → FastAPI server (HF Spaces compatible)
* `Dockerfile` → Deployment

---

## 🚀 Deployment

This project is deployed on **Hugging Face Spaces** using Docker.
The environment runs as a live API and logs structured agent interactions.

---

## 💡 Why This Stands Out

* Real-world healthcare application
* Handles **uncertainty and adversarial cases**
* Encourages **reasoning over shortcuts**
* Robust to API failures
* Fully compliant with OpenEnv specifications

---

## 🏁 Conclusion

This environment provides a strong benchmark for evaluating **decision-making capabilities of LLM agents** in high-stakes, real-world scenarios like healthcare triage.

---

## 👤 Author

Manideep Myakala
