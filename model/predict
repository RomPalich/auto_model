import pandas as pd
import numpy as np
import joblib
import os


class AutoLiquidPredictor:
    """Класс для предсказания цены и срока продажи автомобиля"""

    def __init__(self, models_path='models/'):
        self.models_path = models_path
        self.load_models()

    def load_models(self):
        """Загрузка обученных моделей и энкодеров"""
        self.price_model = joblib.load(os.path.join(self.models_path, 'price_model.pkl'))
        self.days_model = joblib.load(os.path.join(self.models_path, 'days_model.pkl'))
        self.price_features = joblib.load(os.path.join(self.models_path, 'price_features.pkl'))
        self.days_features = joblib.load(os.path.join(self.models_path, 'days_features.pkl'))
        self.encoders = joblib.load(os.path.join(self.models_path, 'encoders.pkl'))

    def prepare_features(self, params):
        """Подготовка признаков из входных параметров"""

        # Кодирование категориальных признаков
        brand_encoded = self.encoders['brand'].transform([params['brand']])[0]
        model_encoded = self.encoders['model'].transform([params['model']])[0]
        color_encoded = self.encoders['color'].transform([params['color']])[0]

        # Вычисление дополнительных признаков
        age = 2025 - params['year']
        mileage_per_year = params['mileage'] / (age + 1)
        is_premium_color = params['color'] in ['Белый', 'Черный', 'Серебристый']

        # Словарь признаков
        features = {
            'brand_encoded': brand_encoded,
            'model_encoded': model_encoded,
            'year': params['year'],
            'mileage': params['mileage'],
            'condition': params['condition'],
            'color_encoded': color_encoded,
            'owners_count': params['owners_count'],
            'accident': params['accident'],
            'age': age,
            'mileage_per_year': mileage_per_year,
            'is_premium_color': int(is_premium_color)
        }

        return features

    def predict_price(self, params):
        """Предсказание цены автомобиля"""
        features = self.prepare_features(params)

        # Создаем DataFrame в правильном порядке признаков
        df = pd.DataFrame([features])[self.price_features]

        price = self.price_model.predict(df)[0]

        # Добавляем доверительный интервал (упрощенно)
        confidence = 0.85 + (params['condition'] - 5) * 0.02
        confidence = min(0.98, max(0.70, confidence))

        # Влияние ДТП на уверенность
        if params['accident']:
            confidence *= 0.9

        price_lower = price * (1 - (1 - confidence) * 0.5)
        price_upper = price * (1 + (1 - confidence) * 0.5)

        return {
            'price': int(price),
            'price_lower': int(price_lower),
            'price_upper': int(price_upper),
            'confidence': confidence
        }

    def predict_days_to_sell(self, params, price_prediction):
        """Предсказание срока продажи"""
        features = self.prepare_features(params)

        # Добавляем предсказанную цену и отклонение
        features['predicted_price'] = price_prediction['price']
        features['price_deviation'] = 1.0  # Для нового объекта принимаем 1

        # Создаем DataFrame в правильном порядке
        df = pd.DataFrame([features])[self.days_features]

        days = self.days_model.predict(df)[0]

        # Корректировка на сезонность
        month_effect = 0
        if 3 <= params['month'] <= 5:  # Весна
            month_effect = -0.15
        elif 11 <= params['month'] <= 12:  # Зима
            month_effect = 0.15

        days = days * (1 + month_effect)

        # Корректировка на состояние
        if params['condition'] >= 8:
            days *= 0.8
        elif params['condition'] <= 4:
            days *= 1.3

        days = max(1, min(180, int(days)))

        # Определяем категорию ликвидности
        if days <= 14:
            liquidity = 'Высокая'
            liquidity_desc = 'Продастся очень быстро, возможно, стоит немного поднять цену'
        elif days <= 30:
            liquidity = 'Средняя'
            liquidity_desc = 'Продажа в стандартные сроки, цена адекватна рынку'
        elif days <= 60:
            liquidity = 'Ниже средней'
            liquidity_desc = 'Может потребоваться корректировка цены или улучшение презентации'
        else:
            liquidity = 'Низкая'
            liquidity_desc = 'Неликвидный автомобиль, рекомендуется существенно снизить цену'

        return {
            'days': days,
            'liquidity': liquidity,
            'liquidity_description': liquidity_desc
        }

    def predict(self, params):
        """Полный прогноз: цена + срок продажи"""
        price = self.predict_price(params)
        days = self.predict_days_to_sell(params, price)

        return {
            'price': price,
            'days': days,
            'input_params': params
        }

    def get_feature_importance(self):
        """Получение важности признаков для интерпретации"""
        importance = pd.DataFrame({
            'feature': self.price_features,
            'importance': self.price_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance.head(10).to_dict('records')


# Создание глобального экземпляра
predictor = AutoLiquidPredictor()
