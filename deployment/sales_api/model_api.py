from fastapi import FastAPI
from pydantic import BaseModel

from deployment.sales_api.api_utils import initialize_sales_predictor


predictor = initialize_sales_predictor()

app = FastAPI()


class PredictionRequest(BaseModel):
    family: str
    store_nbr: int


@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    prediction = predictor.make_prediction(request.family, request.store_nbr)
    return prediction
