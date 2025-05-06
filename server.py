from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel
import re
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()  

origins = os.getenv("ALLOWED_ORIGINS", "")

groq = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


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

class ChatRequest(BaseModel):
    message: str


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

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Sends a chat message to Groq API and streams back the response.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, knowledgeable, and cautious AI doctor named DeepDiagnose. "
                    "Your goal is to assist users by discussing their symptoms, suggesting possible conditions based on their description, "
                    "and advising them to seek professional medical care when necessary. "
                    "You must not make a definitive diagnosis or prescribe treatment. "
                    "Always remind the user that your advice does not replace a visit to a licensed healthcare professional. "
                    "Be clear, empathetic, and use simple language unless the user asks for detailed medical explanations."
                ),
            },
            {
                "role": "user",
                "content": chat_request.message,
            },
        ]

        chat_completion = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True, 
        )

        # Improved streaming generator function
        async def event_generator():
            for chunk in chat_completion:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        # Send properly formatted SSE message
                        yield f"data: {content}\n\n"

        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
