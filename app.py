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

# Define the recommendation function
def recommend1(title, cosine_sim = cosine_sim_nlp_df):
    recommended_movies = []
    idx = indices[indices == title].index[0]   
    print(idx)
    score_series = cosine_sim.iloc[idx].sort_values(ascending = False)   
    top_5_indices = list(score_series.iloc[1:6].index)   
    top_5_indices = [int(i) for i in top_5_indices]
    for i in top_5_indices:  
        recommended_movies.append(list(indices)[i])
        
    return recommended_movies

def recommend2(place_name, similarity_data=cosine_sim_df, k=5):
    index = similarity_data.loc[:, place_name].to_numpy().argpartition(range(-1, -k, -1))
    index_np = np.array(index)
    closest_indices = index_np[-1:-(k+2):-1]
    closest = list(similarity_data.columns[closest_indices])
    if place_name in closest:
        closest.remove(place_name)
    rec = (pd.DataFrame(closest, columns=['Place_Name']).head(k)).to_numpy()
    return [r[0] for r in rec]


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