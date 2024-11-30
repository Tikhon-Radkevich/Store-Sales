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


@app.get("/info")
async def get_info():
    family_store_to_model_df = predictor.family_store_to_model_df.reset_index()
    available_families = family_store_to_model_df["family"].unique().tolist()

    family_store_to_model = (
        family_store_to_model_df.groupby("family")
        .apply(lambda group: dict(zip(group["store_nbr"], group["model"])))
        .to_dict()
    )
    response = {
        "families": available_families,
        "family_store_to_model": family_store_to_model,
    }
    return response
