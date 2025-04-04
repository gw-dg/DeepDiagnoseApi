from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel
import re
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import os

load_dotenv()  

origins = os.getenv("ALLOWED_ORIGINS", "")

# Initialize FastAPI
app = FastAPI(title="Deep Diagnose")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load Preprocessed Data
dfx = pd.read_csv("datasets/processed_data.csv")
symptoms = dfx.columns[1:]  # Extract symptom names



df = pd.read_csv("datasets/dataset2.csv")

# Encode Labels
le = LabelEncoder()
le = LabelEncoder()
df["Disease"] = le.fit_transform(df["Disease"])  # Encode diseases
disease_classes = le.classes_  # Stores the disease names in order
# print(label)

def clean_symptoms(symptom_list):
    cleaned_set = set()
    for symptom in symptom_list:
        symptom = symptom.strip() 
        symptom = re.sub(r"\.\d+$", "", symptom)  
        cleaned_set.add(symptom)  
    return sorted(cleaned_set)  

cleaned_symptoms = clean_symptoms(symptoms)

with open("datasets/Random Forest.pkl", "rb") as f:
    model = pickle.load(f)

# (descriptions, precautions, diet, medications)
sd = pd.read_csv("datasets/description.csv")
sp = pd.read_csv("datasets/precautions_df.csv")
d = pd.read_csv("datasets/diets.csv")
md = pd.read_csv("datasets/medications.csv")

# Define Request Model
class SymptomInput(BaseModel):
    symptoms: list[str]  

@app.get("/")
def root():
    return {"message": "Welcome to the DeepDiagnose API! Use /predict to get disease predictions."}

@app.get("/symptoms")
def get_symptoms():
    """Returns a list of valid symptoms."""
    return {"valid_symptoms": cleaned_symptoms}

@app.get("/diseases")
def get_diseases():
    """Returns a list of valid diseases."""
    return {"all_diseases": disease_classes.tolist()}

@app.post("/predict")
def predict_disease(user_input: SymptomInput):

    # Normalize user symptoms
    user_symptoms = [s.lower() for s in user_input.symptoms]
    

    # Validate Symptoms
    valid_symptoms = [s for s in user_symptoms if s in symptoms]
    invalid_symptoms = [s for s in user_symptoms if s not in symptoms]

    if not valid_symptoms:
        raise HTTPException(status_code=400, detail="Invalid symptoms. Please use /symptoms to see valid options.")

    # Create Input Vector
    t = pd.Series([0] * len(symptoms), index=symptoms)
    t.loc[valid_symptoms] = 1
    t_array = t.to_numpy().reshape(1, -1)

    try:
        probabilities = model.predict_proba(t_array)[0]
        top5_idx = np.argsort(probabilities)[-5:][::-1]  
        top5_proba = probabilities[top5_idx]
        top5_diseases = [disease_classes[int(i)] for i in top5_idx] 
        predictions = []
        for i in range(5):
            disease_name = top5_diseases[i]

            disease_info = {
                "Disease": str(disease_name),  # Ensure it's a string
                "Probability": float(top5_proba[i]),  # Convert numpy.float64 to Python float
                "Description": str(sd.loc[sd["Disease"] == disease_name, "Description"].values[0]) 
                                if disease_name in sd["Disease"].values else None,
                "Precautions": [str(p) for p in sp.loc[sp["Disease"] == disease_name].values.flatten()[1:]] 
                                if disease_name in sp["Disease"].values else [],
                "Diet": [str(d) for d in d.loc[d["Disease"] == disease_name].values.flatten()[1:]] 
                                if disease_name in d["Disease"].values else [],
                "Medications": [str(m) for m in md.loc[md["Disease"] == disease_name].values.flatten()[1:]] 
                                if disease_name in md["Disease"].values else [],
            }

            predictions.append(disease_info)

        return {
            "valid_symptoms": valid_symptoms,
            "invalid_symptoms": invalid_symptoms,
            "predictions": predictions,
        }

    except Exception as e:
        print("Error occurred:", str(e)) 
        raise HTTPException(status_code=500, detail=str(e))
