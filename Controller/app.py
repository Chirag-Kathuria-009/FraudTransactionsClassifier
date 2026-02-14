#Lets get started
from fastapi import FastAPI
import lightgbm as lgb
import pandas as pd 
import shap
import matplotlib.pyplot as plt
import io
import base64


app = FastAPI(title='Fraud Detection API')

model = lgb.Booster(model_file='fraud_datamodel_ieee.txt')

explainer = shap.TreeExplainer(model)


# creating routing

@app.get('/')
def health():
    return {'status':'running'}

@app.post('/predict')
def predict(payload: dict):
    df = pd.DataFrame([payload])
    pred = model.predict(df)[0]
    return {'fraud_probability':float(pred)}

@app.post('/predict_with_explain')
def predict_with_explain(payload: dict):
    df = pd.DataFrame([payload])
    prob = model.predict(df)[0]
    shap_values = explainer(df).values[0]
    shap_dict = dict(zip(df.columns,map(float,shap_values)))
    
    
    
    # Sort by absolute impact (strongest contributors first)
    sorted_importance = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Convert to readable format
    feature_impact_ranked = [
        {"feature": k, "impact": v} for k, v in sorted_importance
    ]

    return {
        "fraud_probability": prob,
        "top_important_features": feature_impact_ranked[:5],  # top 5
        "full_shap": shap_dict
    }
    
@app.post("/predict_image")
def predict_image(payload: dict):
    df = pd.DataFrame([payload])

    prob = float(model.predict(df)[0])
    shap_values = explainer(df).values[0]

    s = pd.Series(shap_values, index=df.columns)
    s = s.reindex(s.abs().sort_values(ascending=False).index)

    # Plot
    plt.figure(figsize=(6,6))
    s.plot(kind="barh")
    plt.title(f"Feature Impact (Fraud Probability = {prob:.2f})")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "fraud_probability": prob,
        "shap_plot_base64_png": img_b64
    }



#handler = Mangum(app)
