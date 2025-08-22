from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
import numpy as np

# Create dummy training data
np.random.seed(42)
data = pd.DataFrame({
    'revenue_growth': np.random.rand(100) * 0.5,
    'debt_to_equity': np.random.rand(100) * 2,
    'news_sentiment': np.random.randint(-5, 6, 100),
    'github_stars': np.random.rand(100) * 100000,
})

# Target credit scores (y) based on features
data['credit_score'] = 700 + data['revenue_growth'] * 100 - data['debt_to_equity'] * 50 + data['news_sentiment'] * 10 + data['github_stars'] * 0.01

X = data[['revenue_growth', 'debt_to_equity', 'news_sentiment', 'github_stars']]
y = data['credit_score']

# Train a simple model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'credit_score_model.pkl')
print("Model trained and saved to 'credit_score_model.pkl'")
