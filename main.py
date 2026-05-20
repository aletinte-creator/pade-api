from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "PADE API funcionando"}

@app.get("/ping")
def ping():
    return {"status": "ok"}
``
