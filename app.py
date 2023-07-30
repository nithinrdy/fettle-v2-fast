from fastapi import FastAPI
import json
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from fastapi import responses


df = pd.read_csv("diseases__encoded.csv").reset_index()
model = SentenceTransformer("all-MiniLM-L12-v2")

app = FastAPI()


@app.get("/data")
def data():
    to_client = df.drop(["symptomVectors", "symptoms", "index", "level_0"], axis=1)
    k = to_client.to_json(orient="records")
    return responses.JSONResponse(content=json.loads(k))


@app.get("/query")
async def query(query: str):
    # Encode the query string
    vecs = model.encode(query)  # Generates vector for query string

    # Calculates cosine similarity between query and all diseases' symptoms; also factors
    # in the rarity of the disease.
    def calculateSimilarity(preVec, rScore):
        cosSim = cosine_similarity([vecs], [np.array(literal_eval(preVec))])[0][0]
        return cosSim + ((4 - rScore) / 20)

    # Drop unnecessary columns and add a new column containing the similarity score
    to_client = df.drop(
        [
            "symptoms",
            "index",
            "level_0",
            "primary_description",
            "secondary_description",
            "rarity",
            "symptom_possibility",
            "raw_symptoms",
        ],
        axis=1,
    )
    to_client["simScore"] = to_client.apply(
        lambda x: calculateSimilarity((x["symptomVectors"]), x["rarityScore"]),
        axis=1,
    )

    # JSONify the dataframe and return it
    return responses.JSONResponse(
        content=json.loads(
            to_client.drop(["symptomVectors", "rarityScore"], axis=1)
            .sort_values(by="simScore", ascending=False)
            .to_json(orient="records")
        )
    )
