from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from model.predict import predictor
import os

app = Flask(__name__)

# Словари для отображения русских названий
BRANDS = [
    'Toyota', 'Hyundai', 'Kia', 'Lada', 'Volkswagen',
    'BMW', 'Mercedes', 'Nissan', 'Renault', 'Skoda'
]

MODELS_BY_BRAND = {
    'Toyota': ['Camry', 'Corolla', 'RAV4', 'Land Cruiser'],
    'Hyundai': ['Solaris', 'Creta', 'Santa Fe', 'Tucson'],
    'Kia': ['Rio', 'Sportage', 'Sorento', 'K5'],
    'Lada': ['Vesta', 'Granta', 'Largus', 'Niva'],
    'Volkswagen': ['Polo', 'Tiguan', 'Passat', 'Jetta'],
    'BMW': ['X5', 'X3', '5 Series', '3 Series'],
    'Mercedes': ['E-Class', 'GLE', 'C-Class', 'GLC'],
    'Nissan': ['Qashqai', 'X-Trail', 'Almera', 'Juke'],
    'Renault': ['Duster', 'Logan', 'Sandero', 'Kaptur'],
    'Skoda': ['Octavia', 'Rapid', 'Kodiaq', 'Karoq']
}

COLORS = ['Белый', 'Черный', 'Серебристый', 'Красный', 'Синий', 'Зеленый']
MONTHS = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html',
                           brands=BRANDS,
                           models=MODELS_BY_BRAND,
                           colors=COLORS,
                           months=MONTHS)


@app.route('/get_models/<brand>')
def get_models(brand):
    """API для получения моделей по марке"""
    models = MODELS_BY_BRAND.get(brand, [])
    return jsonify(models)


@app.route('/predict', methods=['POST'])
def predict():
    """API для предсказания"""
    try:
        data = request.form

        # Сбор параметров
        params = {
            'brand': data.get('brand'),
            'model': data.get('model'),
            'year': int(data.get('year')),
            'mileage': int(data.get('mileage')),
            'condition': int(data.get('condition')),
            'color': data.get('color'),
            'owners_count': int(data.get('owners_count')),
            'accident': int(data.get('accident', 0)),
            'month': int(data.get('month'))
        }

        # Проверка корректности
        if params['year'] < 1990 or params['year'] > 2025:
            return jsonify({'error': 'Некорректный год выпуска'}), 400

        if params['mileage'] < 0 or params['mileage'] > 500000:
            return jsonify({'error': 'Некорректный пробег'}), 400

        # Предсказание
        result = predictor.predict(params)

        # Форматирование результата
        response = {
            'price': result['price']['price'],
            'price_lower': result['price']['price_lower'],
            'price_upper': result['price']['price_upper'],
            'confidence': f"{result['price']['confidence'] * 100:.0f}",
            'days_to_sell': result['days']['days'],
            'liquidity': result['days']['liquidity'],
            'liquidity_description': result['days']['liquidity_description']
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feature_importance')
def feature_importance():
    """API для получения важности признаков"""
    importance = predictor.get_feature_importance()
    return jsonify(importance)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)