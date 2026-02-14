import bentoml 
from bentoml.io import JSON 
import lightgbm as lgb 
import pandas as pd
import shap 

from bentoml import Service
import inspect

import os
print("BENTOML_SDK =", os.getenv("BENTOML_SDK"))


print(bentoml.__version__)
# Load your trained model
model = lgb.Booster(model_file='fraud_datamodel_ieee.txt')  # Your saved model file

bentoml.lightgbm.save_model("fraud_detector", model)

runner = bentoml.lightgbm.get("fraud_detector:latest").to_runner()
explainer = shap.TreeExplainer(model)  # For optional SHAP

svc = bentoml.Service("fraud_detection", runners=[runner])
@svc.api(input=JSON(), output=JSON())
async def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    prob = (await runner.predict.run(df))[0]  # Fraud probability
    response = {"fraud_probability": float(prob)}

    # Optional SHAP (only if requested)
    if input_data.get("explain", False):
        shap_values = explainer(df).values[0]
        response["shap_explanation"] = dict(zip(df.columns, shap_values))

    return response

