# Анализ мошеннических транзакций - Полный код для Spyder

# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

print("=== ЗАГРУЗКА ДАННЫХ ===")
# Загрузка данных
df = pd.read_parquet(r'C:\Users\Ольга\Desktop\Fraud\transaction_fraud_data.parquet', engine='pyarrow')
print(f"Данные успешно загружены. Размер: {df.shape}")

print("\n=== БАЗОВЫЙ АНАЛИЗ ДАННЫХ ===")
print(f"Размер данных: {df.shape}")
print(f"Процент мошеннических операций: {df['is_fraud'].mean()*100:.2f}%")
print("\nТипы данных:")
print(df.dtypes)
print("\nПропущенные значения:")
print(df.isnull().sum())

print("\n=== ВРЕМЕННОЙ АНАЛИЗ ===")
# Создание временных признаков
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['timestamp'].dt.dayofweek > 4

# Анализ по часам
fraud_by_hour = df.groupby('hour')['is_fraud'].mean()
print("Доля мошенничества по часам:")
print(fraud_by_hour.round(4))

# Анализ по дням недели
fraud_by_day = df.groupby('day_of_week')['is_fraud'].mean()
days_names = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
fraud_by_day.index = [days_names[i] for i in fraud_by_day.index]
print("\nДоля мошенничества по дням недели:")
print(fraud_by_day.round(4))

# Сравнение выходные/будни
weekend_fraud = df.groupby('is_weekend')['is_fraud'].mean()
print("\nСравнение выходные/будни:")
print(f"Будни: {weekend_fraud[False]:.4f}")
print(f"Выходные: {weekend_fraud[True]:.4f}")

print("\n=== ФИНАНСОВЫЙ АНАЛИЗ ===")
legit_amounts = df[df['is_fraud'] == False]['amount']
fraud_amounts = df[df['is_fraud'] == True]['amount']
print(f"Средняя сумма легальных операций: {legit_amounts.mean():.2f}")
print(f"Средняя сумма мошеннических операций: {fraud_amounts.mean():.2f}")
print(f"Отношение сумм: {fraud_amounts.mean()/legit_amounts.mean():.1f}x")

print("\n=== ГЕОГРАФИЧЕСКИЙ АНАЛИЗ ===")
fraud_by_country = df.groupby('country')['is_fraud'].agg(['count', 'mean'])
fraud_by_country = fraud_by_country.sort_values('mean', ascending=False)

print("Топ-10 стран по доле мошенничества:")
print(fraud_by_country.head(10).round(4))

print("\nТоп-10 стран по количеству мошеннических операций:")
print(fraud_by_country.sort_values('count', ascending=False).head(10).round(4))

print("\n=== АНАЛИЗ КАТЕГОРИЙ ВЕНДОРОВ ===")
if 'vendor_category' in df.columns:
    risk_by_category = df.groupby('vendor_category')['is_fraud'].agg(['count', 'mean'])
    risk_by_category = risk_by_category.sort_values('mean', ascending=False)
    print("Топ-10 категорий по доле мошенничества:")
    print(risk_by_category.head(10).round(4))
else:
    print("Колонка 'vendor_category' отсутствует в данных")

print("\n=== АНАЛИЗ ТИПОВ ВЕНДОРОВ ===")
if 'vendor_type' in df.columns:
    risk_by_type = df.groupby('vendor_type')['is_fraud'].agg(['count', 'mean'])
    risk_by_type = risk_by_type.sort_values('mean', ascending=False)
    print("Топ-10 типов вендоров по доле мошенничества:")
    print(risk_by_type.head(10).round(4))
else:
    print("Колонка 'vendor_type' отсутствует в данных")

print("\n=== АНАЛИЗ ПЛАТЕЖНЫХ ИНСТРУМЕНТОВ ===")
if 'card_type' in df.columns:
    risk_by_card = df.groupby('card_type')['is_fraud'].agg(['count', 'mean'])
    risk_by_card = risk_by_card.sort_values('mean', ascending=False)
    print("Риск по типам карт:")
    print(risk_by_card.round(4))
else:
    print("Колонка 'card_type' отсутствует в данных")

print("\n=== АНАЛИЗ ПРИСУТСТВИЯ КАРТЫ ===")
if 'is_card_present' in df.columns:
    risk_by_presence = df.groupby('is_card_present')['is_fraud'].mean()
    print("Риск по присутствию карты:")
    print(risk_by_presence.round(4))
else:
    print("Колонка 'is_card_present' отсутствует в данных")

print("\n=== АНАЛИЗ КАНАЛОВ И УСТРОЙСТВ ===")
if 'channel' in df.columns:
    risk_by_channel = df.groupby('channel')['is_fraud'].agg(['count', 'mean'])
    risk_by_channel = risk_by_channel.sort_values('mean', ascending=False)
    print("Риск по каналам:")
    print(risk_by_channel.round(4))
else:
    print("Колонка 'channel' отсутствует в данных")

# ВИЗУАЛИЗАЦИЯ
print("\n=== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ===")

plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Доля мошенничества по часам
fraud_by_hour.plot(kind='bar', alpha=0.7, ax=axes[0, 0], color='red')
axes[0, 0].set_title('Доля мошенничества по часам')
axes[0, 0].set_xlabel('Час дня')
axes[0, 0].set_ylabel('Доля мошенничества')
axes[0, 0].grid(True, alpha=0.3)

# 2. Топ-10 стран
fraud_by_country['mean'].head(10).plot(kind='bar', alpha=0.7, ax=axes[0, 1], color='blue')
axes[0, 1].set_title('Топ-10 стран по доле мошенничества')
axes[0, 1].set_xlabel('Страна')
axes[0, 1].set_ylabel('Доля мошенничества')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# 3. Сравнение сумм
axes[1, 0].bar(['Легальные', 'Мошеннические'],
               [legit_amounts.mean(), fraud_amounts.mean()],
               color=['green', 'orange'], alpha=0.7)

axes[1, 0].set_title('Сравнение средних сумм транзакций')
axes[1, 0].set_ylabel('Сумма')
axes[1, 0].tick_params(axis='x', rotation=0)
axes[1, 0].grid(True, alpha=0.3)

# 4. Дни недели
fraud_by_day.plot(kind='bar', alpha=0.7, ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Доля мошенничества по дням недели')
axes[1, 1].set_xlabel('День недели')
axes[1, 1].set_ylabel('Доля мошенничества')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()