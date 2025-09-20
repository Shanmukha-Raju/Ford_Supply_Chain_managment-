import requests
from pytrends.request import TrendReq
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
OPENWEATHER_API_KEY = "df05624eaeb076ee9af6588294e4c532"
ml_model = None
scaler = StandardScaler()
is_model_trained = False
def get_google_trend_score(keyword="F-150"):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe='now 1-H')
        trends = pytrends.interest_over_time()
        if not trends.empty:
            return int(trends[keyword].iloc[-1])
    except:
        pass
    return 50
def get_live_temperature(city="Detroit"):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=imperial"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()["main"]["temp"]
    except:
        pass
    return 70
def train_ml_model():
    global ml_model, scaler, is_model_trained
    print(" Training ML model with simulated historical data...")
    np.random.seed(42)
    n_samples = 1000
    months = np.random.randint(1, 13, n_samples)
    trends = np.random.randint(30, 100, n_samples)
    temps = np.random.randint(20, 100, n_samples)
    risks = np.random.uniform(0.1, 0.9, n_samples)
    X = np.column_stack([months, trends, temps, risks])
    y = 8000 + trends * 40 + (temps - 70) * 15 - risks * 3000 + np.random.normal(0, 300, n_samples)
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Swap to LinearRegression() if you want
    model.fit(X_scaled, y)
    ml_model = model
    is_model_trained = True
    print(" ML model trained and ready!")
def predict_demand_ml(month, trend_score, temperature, supplier_risk=0.5):
    global ml_model, scaler, is_model_trained
    if not is_model_trained:
        return 10000
    try:
        features = np.array([[month, trend_score, temperature, supplier_risk]])
        features_scaled = scaler.transform(features)
        prediction = ml_model.predict(features_scaled)[0]
        return int(max(5000, min(prediction, 30000)))
    except:
        return 10000
def get_action(trend_score, part_name):
    if trend_score > 80:
        return f" TRENDING! Ramp up {part_name} production!"
    elif trend_score < 30:
        return f" LOW INTEREST. Reduce {part_name} orders."
    else:
        return f" Normal demand for {part_name}. Maintain stock."
def ford_ai_simple(part="F-150", city="Detroit"):
    global is_model_trained

    if not is_model_trained:
        train_ml_model()
    trend_score = get_google_trend_score(part)
    temperature = get_live_temperature(city)
    month = datetime.now().month

    demand = predict_demand_ml(month, trend_score, temperature, supplier_risk=0.3)
    action = get_action(trend_score, part)
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "part": part,
        "predicted_demand": demand,
        "recommended_action": action,
        "data_sources": {
            "google_trends": f"{trend_score}/100",
            "temperature_f": f"{temperature:.1f}°F in {city}",
            "ml_model": "RandomForestRegressor"
        }
    }
if __name__ == "__main__":
    print(" Welcome to FORD SUPPLY CHAIN AI — ML + INTERACTIVE MODE")
    print(" Powered by Google Trends + Weather API + Machine Learning")
    print(" Type 'quit' or 'stop' anytime to exit.\n")
    if not is_model_trained:
        train_ml_model()
    while True:
        part = input(" Enter vehicle part/model (e.g., F-150, Mustang, Bronco): ").strip()
        if part.lower() in ["quit", "stop", "exit"]:
            print("\n Exiting... Thank you for using Ford AI!")
            break
        city = input(" Enter city for weather (e.g., Detroit, Chicago, Phoenix): ").strip()
        if city.lower() in ["quit", "stop", "exit"]:
            print("\n Exiting... Thank you for using Ford AI!")
            break
        print(f"\n Analyzing demand for '{part}' in '{city}'...")
        print("-" * 60)
        result = ford_ai_simple(part, city)
        print(f"⏱{result['timestamp']}")
        print(f" Part: {result['part']}")
        print(f" Predicted Demand: {result['predicted_demand']:,} units")
        print(f" Action: {result['recommended_action']}")
        print(f" ML Model: {result['data_sources']['ml_model']}")
        print(f" Data: Trends={result['data_sources']['google_trends']}, Temp={result['data_sources']['temperature_f']}")
        print("-" * 60 + "\n")
        print(" ANALYZING TOP 10 GLOBAL CITIES BY DEMAND FOR '{}'...".format(part))
        print("=" * 70)
        global_cities = [
            "New York", "London", "Tokyo", "Los Angeles", "Paris",
            "Dubai", "Singapore", "Sydney", "Toronto", "Berlin",
            "Mumbai", "Sao Paulo", "Mexico City", "Shanghai", "Moscow",
            "Seoul", "Bangkok", "Istanbul", "Rome", "Cape Town"
        ]
        city_predictions = []
        current_month = datetime.now().month
        for c in global_cities:
            try:
                trend = get_google_trend_score(part)
                temp = get_live_temperature(c)
                demand = predict_demand_ml(current_month, trend, temp, 0.3)
                city_predictions.append((c, demand, trend, temp))
            except:
                continue  
        city_predictions.sort(key=lambda x: x[1], reverse=True)

        print(f"{'Rank':<5} {'City':<15} {'Demand':<12} {'Trend':<8} {'Temp(°F)':<10}")
        print("-" * 70)
        for rank, (city, demand, trend, temp) in enumerate(city_predictions[:10], 1):
            print(f"{rank:<5} {city:<15} {demand:<12,} {trend:<8} {temp:<10.1f}")
        if city_predictions:
            top_city, top_demand, _, _ = city_predictions[0]
            print(f"\n HIGHEST GLOBAL DEMAND: {top_city} → {top_demand:,} units")
        print("=" * 70 + "\n")
