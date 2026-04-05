🏥 Medical Triage OpenEnv Environment

 📌 Overview

This project implements a **real-world medical triage simulation environment** designed for evaluating AI agents.
The agent must analyze patient symptoms, ask relevant questions, and make critical healthcare decisions under uncertainty.

Unlike toy environments, this system models **real decision-making challenges** including incomplete information, time pressure, and risk-sensitive outcomes.



 🎯 Objectives

* Simulate realistic patient triage scenarios
* Evaluate agent reasoning under uncertainty
* Encourage safe and optimal medical decision-making
* Provide meaningful reward signals with partial progress



 🧠 Environment Design

 🔍 Observation Space

Each step returns structured patient data:

* Symptoms (list of strings)
* Age
* Known conditions
* Vitals (hidden initially)
* Patient responses
* Time elapsed
* Available actions



 ⚙️ Action Space

The agent can:

* Ask questions:

  * `ask_symptom_details`
  * `ask_vitals`
  * `ask_history`
* Make decisions:

  * `send_to_ER`
  * `schedule_doctor`
  * `prescribe_basic_meds`
  * `ignore_case`



 🏆 Reward Function

* +1.0 → Correct decision
* -1.0 → Critical mistake (e.g., ignoring emergency)
* -0.4 → Incorrect non-critical decision
* +0.1 → Useful question
* +0.03 → Less useful question
* -0.05 → Repeated question
* Time penalty applied in hard scenarios



 📊 Tasks

🟢 Easy

* Clear symptoms
* Low-risk cases
* Straightforward decisions

 🟡 Medium

* Moderate ambiguity
* Requires reasoning through multiple signals

 🔴 Hard

* High-risk cases (e.g., heart attack disguised as mild symptoms)
* Requires fast and accurate decisions
* Includes time penalty



 🧪 Example Run

```
[START] task=medical-triage env=medtriage-env model=baseline-agent
[STEP] step=1 action=ask_vitals reward=0.10 done=false error=null
[STEP] step=2 action=send_to_ER reward=1.00 done=true error=null
[END] success=true steps=2 rewards=0.10,1.00
```

---

 🏗️ Project Structure

```
env/
  env.py
  models.py

evaluation/
  graders.py

inference.py
Dockerfile
requirements.txt
openenv.yaml
README.md
```

---

 ⚙️ Setup Instructions

 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

 2️⃣ Run locally

```
python inference.py
```

---

 🐳 Docker Usage

 Build image

```
docker build -t med-triage-env .
```

 Run container

```
docker run med-triage-env
```

---

 📈 Evaluation

The environment includes deterministic graders to evaluate:

* Decision accuracy
* Question usefulness
* Efficiency (steps taken)
* Safety (penalties for critical mistakes)

Scores are normalized between **0.0 and 1.0**.


 🚀 Key Features

 ✅ Real-world medical decision simulation
 ✅ Hidden ground truth (agent must infer)
 ✅ Multi-step reasoning environment
 ✅ Safety-critical reward design
 ✅ Time pressure in complex scenarios
 ✅ Fully Dockerized & reproducible



 🏁 Conclusion

This environment provides a **robust benchmark for evaluating AI agents in high-stakes decision-making scenarios**, emphasizing correctness, efficiency, and safety.

---



Manideep
