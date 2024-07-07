from fastapi import FastAPI
import pandas as pd
import uvicorn
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load the cosine similarity matrix and indices
cosine_sim_nlp_df = pd.read_csv('data_processed/cosine_sim_nlp.csv')
cosine_sim_df=pd.read_csv('data_processed/cosine_sim.csv')
indices = pd.read_csv('data_processed/indices.csv')
indices = pd.Series(indices['Place_Name'])

# Define the request body model
class PlaceName(BaseModel):
    place_name: str

# Define the FastAPI route to recommend similar places
@app.post("/recommend1/")
async def get_recommendations(place_name_data: PlaceName):
    place_name = place_name_data.place_name
    recommended_places = recommend1(place_name)
    return {"recommended_places": recommended_places}

@app.post("/recommend2/")
async def get_recommendations(place_name_data: PlaceName):
    place_name = place_name_data.place_name
    recommended_places = recommend2(place_name)
    return {"recommended_places": recommended_places}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
