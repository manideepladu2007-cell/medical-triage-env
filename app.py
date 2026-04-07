from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.on_event("startup")
def run_inference():
    subprocess.Popen(["python", "inference.py"])
