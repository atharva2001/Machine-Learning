import pickle as pkl
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

model = pkl.load(open('model.pkl', 'rb')) 

app = FastAPI(title="Fraud Detection")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Hello World"}


class Fraud(BaseModel):
    amount: float = 9839.64
    oldbalanceOrg: float = 9839.64
    newbalanceOrig: float = 0.0
    oldbalanceDest: float = 0.0
    newbalanceDest: float = 0.0
    isFlaggedFraud: int = 0

@app.post("/predict")
def predict(body: Fraud):
    res = model.predict([[body.amount, body.oldbalanceOrg, body.newbalanceOrig, body.oldbalanceDest, body.newbalanceDest, body.isFlaggedFraud]])
    return {"fraud":  bool(res[0])}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)