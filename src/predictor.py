import pandas as pd
from logger import get_logger

logger = get_logger(__name__)

def predict_cluster(model, age, income, avg_score):
    logger.info(f"Predicting cluster for Age: {age}, Income: {income}")
    input_df = pd.DataFrame([[age, income, avg_score]], columns=["Age", "Annual_Income", "Spending_Score"])
    prediction = model.predict(input_df)[0]
    logger.info(f"Predicted cluster: {prediction}")
    return prediction