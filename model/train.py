import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os


def train_price_model(df):
    """Обучение модели для предсказания цены"""

    # Признаки для модели цены
    feature_cols = [
        'brand_encoded', 'model_encoded', 'year', 'mileage',
        'condition', 'color_encoded', 'owners_count', 'accident',
        'age', 'mileage_per_year', 'is_premium_color'
    ]

    X = df[feature_cols]
    y = df['price']

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучение XGBoost
    print("Обучение XGBoost для цены...")
    price_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    price_model.fit(X_train, y_train)

    # Оценка
    y_pred = price_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Модель цены - MAE: {mae:,.0f} руб, RMSE: {rmse:,.0f} руб, R2: {r2:.3f}")

    return price_model, feature_cols


def train_days_model(df, price_model, feature_cols):
    """Обучение модели для предсказания срока продажи"""

    # Получаем предсказанные цены для всех данных
    X_price = df[feature_cols]
    predicted_prices = price_model.predict(X_price)

    # Добавляем предсказанную цену как признак
    df['predicted_price'] = predicted_prices
    df['price_deviation'] = df['price'] / df['predicted_price']

    # Признаки для модели срока продажи
    days_feature_cols = feature_cols + ['predicted_price', 'price_deviation']

    X = df[days_feature_cols]
    y = df['days_to_sell']

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Обучение Random Forest для срока продажи
    print("Обучение Random Forest для срока продажи...")
    days_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    days_model.fit(X_train, y_train)

    # Оценка
    y_pred = days_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Модель срока - MAE: {mae:.1f} дней, RMSE: {rmse:.1f} дней, R2: {r2:.3f}")

    return days_model, days_feature_cols


def save_models(price_model, days_model, price_features, days_features):
    """Сохранение моделей"""
    os.makedirs('../models', exist_ok=True)

    joblib.dump(price_model, '../models/price_model.pkl')
    joblib.dump(days_model, '../models/days_model.pkl')
    joblib.dump(price_features, '../models/price_features.pkl')
    joblib.dump(days_features, '../models/days_features.pkl')

    print("Модели сохранены в папку models/")


if __name__ == '__main__':
    # Загрузка данных
    df = pd.read_csv('../data/car_data.csv')

    # Обучение моделей
    price_model, price_features = train_price_model(df)
    days_model, days_features = train_days_model(df, price_model, price_features)

    # Сохранение
    save_models(price_model, days_model, price_features, days_features)