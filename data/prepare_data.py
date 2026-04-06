import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def generate_synthetic_data(n_samples=15000):
    """
    Генерация данных на основе реальных отчетов Авто.ру (2026 год)

    Источники:
    - Авто.ру: средняя цена иномарок с пробегом 2,28 млн руб (январь 2026) [citation:2][citation:10]
    - Авто.ру: средняя цена российских авто 723 тыс руб (январь 2026) [citation:2][citation:10]
    - Авто.ру: средняя цена китайских авто 2,12 млн руб (январь 2026) [citation:2][citation:10]
    - Авто.ру Бизнес: новые российские авто 1,99 млн руб (февраль 2026) [citation:6]
    - Авто.ру Бизнес: доля качественных авто у дилеров 81-83% [citation:3]
    - Авто.ру: предложение авто до 15 лет сократилось на 13% в январе [citation:2][citation:10]
    """
    np.random.seed(42)

    # Марки и модели с привязкой к реальным ценовым данным Авто.ру
    brands = {
        # Премиальные и массовые иномарки (Toyota, BMW, Mercedes, Hyundai, Kia и др.)
        # Средняя цена 2,28 млн руб (иномарки с пробегом) [citation:2][citation:10]
        # ИНОМАРКИ (реально ~2.2–2.5 млн)
    'Toyota': {'models': ['Camry', 'Corolla', 'RAV4'], 'base_price': 2_400_000},
    'BMW': {'models': ['X5', 'X3', '3 Series'], 'base_price': 2_600_000},
    'Mercedes': {'models': ['E-Class', 'C-Class'], 'base_price': 2_700_000},
    'Hyundai': {'models': ['Solaris', 'Creta'], 'base_price': 2_100_000},
    'Kia': {'models': ['Rio', 'Sportage'], 'base_price': 2_050_000},
    'Volkswagen': {'models': ['Polo', 'Tiguan'], 'base_price': 2_200_000},

    # КИТАЙСКИЕ (≈ 2.0–2.3 млн)
    'Chery': {'models': ['Tiggo 4', 'Tiggo 7'], 'base_price': 2_100_000},
    'Geely': {'models': ['Coolray', 'Atlas'], 'base_price': 2_200_000},
    'Haval': {'models': ['Jolion', 'F7'], 'base_price': 2_150_000},

    # РОССИЙСКИЕ (≈ 700–900 тыс)
    'Lada': {'models': ['Vesta', 'Granta'], 'base_price': 800_000},
    'UAZ': {'models': ['Patriot'], 'base_price': 900_000}
    }

    # Цены на отдельные модели из отчетов Авто.ру (январь 2026) [citation:2][citation:10]
    model_price_adjustments = {
        'Haval Jolion': 0.75,  # 1,59 млн вместо 2,12 млн (скидка 25%)
        'Chery Tiggo 4': 0.56,  # 1,19 млн (скидка 44%)
        'Changan Alsvin': 0.46,  # 968 тыс (скидка 54%)
    }

    # Доля качественных автомобилей по данным Авто.ру Бизнес [citation:3]
    # У дилеров и перекупщиков 81-83% качественных авто, у частников 72%
    seller_type_probs = {
        'dealer': 0.82,  # 82% качественных авто у дилеров
        'professional': 0.83,  # 83% у профессиональных перекупщиков
        'private': 0.72  # 72% у частников
    }

    data = []

    for _ in range(n_samples):
        # Выбор марки
        brand = np.random.choice(list(brands.keys()))
        brand_info = brands[brand]
        model = np.random.choice(brand_info['models'])
        segment = brand_info['segment']

        # Базовая цена из данных Авто.ру
        base_price = brand_info['base_price']

        # Корректировка цены для конкретных моделей (по данным Авто.ру)
        model_key = f"{brand} {model}"
        if model_key in model_price_adjustments:
            base_price = int(base_price * model_price_adjustments[model_key])

        # Возраст автомобиля (1-15 лет, так как 80% предложения — авто до 15 лет [citation:2])
        year = np.random.randint(2011, 2026)
        age = 2026 - year

        # Пробег (реалистичные значения: 10 000 - 250 000 км)
        # Среднегодовой пробег в России ~17 000 км
        mileage = int(np.random.normal(age * 17000, age * 5000))
        mileage = max(5000, min(350000, mileage))

        # Состояние (1-10 баллов)
        # Для дилеров/перекупщиков выше вероятность хорошего состояния [citation:3]
        condition = np.random.randint(1, 11)

        # Цвет
        color = np.random.choice(['Белый', 'Черный', 'Серебристый', 'Красный', 'Синий', 'Серый'])

        # Количество владельцев (реалистично: 1-4)
        owners_count = np.random.randint(1, min(5, age // 4 + 2))

        # Наличие ДТП (по статистике ~25-30% авто имеют ДТП в истории)
        accident = np.random.choice([0, 1], p=[0.7, 0.3])

        # Месяц продажи (сезонность: весной спрос выше)
        month = np.random.randint(1, 13)

        # --- РАСЧЕТ ЦЕНЫ НА ОСНОВЕ ДАННЫХ АВТО.РУ ---

        # Корректировка по возрасту (падение стоимости ~10-12% в год)
        # Амортизация (реально сейчас медленнее падает цена)
        age_factor = 0.92 ** age

        price = base_price * age_factor

        # Пробег (сильнее влияет)
        price *= max(0.6, 1 - mileage / 300000)

        # Состояние (важнее, чем раньше)
        price *= (0.7 + condition / 15)

        # Владельцы
        price *= max(0.85, 1 - (owners_count - 1) * 0.02)

        # ДТП (сильнее влияет)
        if accident:
            price *= 0.80

        # Рыночный шум
        price *= np.random.uniform(0.9, 1.1)

        # Корректировка по пробегу
        if mileage > 100000:
            mileage_penalty = 1 - (mileage - 100000) / 400000
            mileage_penalty = max(0.65, mileage_penalty)
            price *= mileage_penalty

        # Корректировка по состоянию (от 0.65 до 1.15)
        condition_factor = 0.65 + condition / 18
        price *= condition_factor

        # Премиальные цвета (белый, черный, серебристый) +3%
        premium_colors = ['Белый', 'Черный', 'Серебристый', 'Серый']
        if color in premium_colors:
            price *= 1.03

        # Каждый дополнительный владелец снижает цену на ~1.5%
        owners_factor = 1 - (owners_count - 1) * 0.015
        owners_factor = max(0.90, owners_factor)
        price *= owners_factor

        # ДТП: снижение на 12-18% (по данным отчетов Авто.ру)
        if accident:
            accident_penalty = 0.85 if condition > 6 else 0.80
            price *= accident_penalty

        # Рыночная вариативность (±7%)
        price *= np.random.uniform(0.93, 1.07)

        # Округление до тысяч
        price = int(round(price / 1000) * 1000)

        # --- РАСЧЕТ СРОКА ПРОДАЖИ НА ОСНОВЕ ДАННЫХ АВТО.РУ ---
        # По данным Авто.ру, средний срок экспозиции:
        # - У дилеров: 30-45 дней
        # - У частников: 45-90 дней
        # - Весной быстрее, зимой медленнее
        if condition >= 8:
            days_to_sell -= 10
        elif condition <= 4:
            days_to_sell += 20

        # ДТП
        if accident:
            days_to_sell += 20

        # Сезонность (сейчас сильнее влияет)
        if 3 <= month <= 5:
            days_to_sell -= 10
        elif 11 <= month <= 2:
            days_to_sell += 15

        # Цена относительно рынка
        if price > base_price:
            days_to_sell += 10

        # Ограничение
        days_to_sell = int(max(3, min(180, days_to_sell)))
        # Базовый срок продажи (дней)
        if segment == 'chinese':
            # Китайские авто продаются чуть медленнее из-за насыщения рынка [citation:6]
            days_to_sell = 55
        elif segment == 'import':
            days_to_sell = 45
        else:
            days_to_sell = 50

        # Состояние
        if condition >= 8:
            days_to_sell -= 12
        elif condition <= 4:
            days_to_sell += 18

        # ДТП замедляет продажу
        if accident:
            days_to_sell += 15

        # Сезонность (весна — высокий спрос, зима — низкий) [citation:2][citation:10]
        if 3 <= month <= 5:  # Весна
            days_to_sell -= 12
        elif 11 <= month <= 12:  # Зима
            days_to_sell += 10
        elif month == 1:  # Январь (затишье)
            days_to_sell += 8

        # Ценовой фактор (слишком высокая цена увеличивает срок)
        # Имитация средней рыночной цены
        market_price = price * np.random.uniform(0.88, 0.98)
        if price > market_price * 1.1:
            days_to_sell += int((price / market_price - 1) * 50)

        # Ограничения и случайный шум
        days_to_sell = max(3, min(180, int(days_to_sell + np.random.normal(0, 8))))

        data.append({
            'brand': brand,
            'model': model,
            'year': year,
            'mileage': mileage,
            'condition': condition,
            'color': color,
            'owners_count': owners_count,
            'accident': accident,
            'month': month,
            'price': price,
            'days_to_sell': days_to_sell
        })

    df = pd.DataFrame(data)
    return df


def prepare_and_save_data():
    """Подготовка данных и сохранение предобработчиков"""

    print("=" * 60)
    print("Генерация данных на основе отчетов Авто.ру (январь-март 2026)")
    print("Источники:")
    print("  - Авто.ру: средняя цена иномарок с пробегом 2,28 млн ₽ [citation:2][citation:10]")
    print("  - Авто.ру: средняя цена российских авто 723 тыс ₽ [citation:2][citation:10]")
    print("  - Авто.ру: средняя цена китайских авто 2,12 млн ₽ [citation:2][citation:10]")
    print("  - Авто.ру Бизнес: цены на новые авто 1,99-3,94 млн ₽ [citation:6]")
    print("  - Авто.ру Бизнес: качество предложения дилеров 81-83% [citation:3]")
    print("=" * 60)

    df = generate_synthetic_data(20000)

    # Кодирование категориальных признаков
    categorical_cols = ['brand', 'model', 'color']
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        encoders[col] = le

    # Создание дополнительных признаков
    df['age'] = 2026 - df['year']
    df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
    df['is_premium_color'] = df['color'].isin(['Белый', 'Черный', 'Серебристый', 'Серый']).astype(int)

    # Признаки для второй модели
    df['price_normalized'] = df['price'] / df.groupby(['brand', 'model'])['price'].transform('mean')

    # Сохранение
    os.makedirs('../models', exist_ok=True)
    df.to_csv('../data/car_data.csv', index=False)
    joblib.dump(encoders, '../models/encoders.pkl')

    print(f"\n✅ Данные сохранены: {len(df)} записей")

    # Статистика по сегментам
    print("\n📊 Статистика цен по сегментам (данные Авто.ру):")

    segment_map = {
        'Toyota': 'Иномарки', 'BMW': 'Иномарки', 'Mercedes-Benz': 'Иномарки',
        'Hyundai': 'Иномарки', 'Kia': 'Иномарки', 'Volkswagen': 'Иномарки',
        'Nissan': 'Иномарки', 'Skoda': 'Иномарки', 'Mazda': 'Иномарки',
        'Chery': 'Китайские', 'Geely': 'Китайские', 'Haval': 'Китайские',
        'Changan': 'Китайские', 'Exeed': 'Китайские',
        'Lada': 'Российские', 'UAZ': 'Российские'
    }
    df['segment'] = df['brand'].map(segment_map)

    stats = df.groupby('segment')['price'].agg(['mean', 'min', 'max']).round(0).astype(int)
    print(stats)

    print(f"\n🏁 Средняя цена всех авто: {df['price'].mean():,.0f} ₽")
    print(f"⏱️ Средний срок продажи: {df['days_to_sell'].mean():.0f} дней")
    print(f"📈 Диапазон цен: {df['price'].min():,.0f} — {df['price'].max():,.0f} ₽")

    # Сравнение с реальными данными
    print("\n🔍 Сравнение с реальными данными Авто.ру:")
    print(f"   Иномарки: модель {stats.loc['Иномарки', 'mean']:,.0f} ₽ | реальные 2 280 000 ₽")
    print(f"   Китайские: модель {stats.loc['Китайские', 'mean']:,.0f} ₽ | реальные 2 120 000 ₽")
    print(f"   Российские: модель {stats.loc['Российские', 'mean']:,.0f} ₽ | реальные 723 000 ₽")

    return df


if __name__ == '__main__':
    prepare_and_save_data()