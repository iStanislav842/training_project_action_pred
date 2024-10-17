import json
import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class Form(BaseModel):
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_model: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object
    

class Prediction(BaseModel):
    event_action: int


with open('data/hit_predict.pkl', 'rb') as file:
    model = dill.load(file)


@app.get('/status')
def status():
    return "I'm OKey"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.model_dump()])
    y = model['model'].predict(df)
    return {
        "event_action": y
    }

#uvicorn main:app --reload
