from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deployment.storesales.api_utils import initialize_sales_predictor


app = FastAPI()

predictor = initialize_sales_predictor()


class PredictionRequest(BaseModel):
    family: str
    store_nbr: int


@app.post("/predict/")
async def make_prediction(request: PredictionRequest):
    try:
        prediction = predictor.make_prediction(request.family, request.store_nbr)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
