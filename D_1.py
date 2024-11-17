import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Пути к файлам
file_path = r'C:\Users\Magom\PYTHON\python_Diplom\sample_dataset.xlsx'
output_file_path = r'C:\Users\Magom\PYTHON\python_Diplom\sample_dataset_updated.xlsx'

# Загрузка данных с использованием Pandas
df = pd.read_excel(file_path)

# 1. Первичный анализ данных
print("Распределение данных (первые 5 строк):")
print(df.head())

print("\nИнформация о данных (информация о столбцах, типах данных и количестве значений):")
print(df.info())

# Статистическое описание данных
print("\nСтатистическое описание данных:")
print(df.describe())

# 2. Работа с Dask и сравнение производительности
# Конвертация данных в формат Dask DataFrame
dask_df = dd.from_pandas(df, npartitions=4)

# Анализ данных с использованием Pandas
start_time = time.time()
pandas_result = df.groupby('Year').agg({
    'Экономический рост': 'mean',
    'Уровень безработицы': 'mean'
}).reset_index()
pandas_time = time.time() - start_time
print(f"\nВремя выполнения для Pandas: {pandas_time:.4f} секунд")

# Анализ данных с использованием Dask
start_time = time.time()
dask_result = dask_df.groupby('Year').agg({
    'Экономический рост': 'mean',
    'Уровень безработицы': 'mean'
}).compute()
dask_time = time.time() - start_time
print(f"\nВремя выполнения для Dask: {dask_time:.4f} секунд")

# 3. Создание нового столбца "общий показатель экономической производительности, учитывающий рост и инфляцию"
if 'Экономический рост' in df.columns and 'Инфляция' in df.columns:
    df['общий показатель экономической производительности, учитывающий рост и инфляцию'] = (
            df['Экономический рост'] - df['Инфляция']
    )
    print("\nДанные с добавленным новым столбцом:")
    print(df.head())

# Сохранение изменённого DataFrame в новый Excel-файл
df.to_excel(output_file_path, index=False)
print(f"\nИзмененные данные сохранены в файл: {output_file_path}")

# 4. Корреляционный анализ
# Фильтруем только числовые столбцы
numeric_data = df.select_dtypes(include=['number'])

# Построение корреляционной матрицы
correlation_matrix = numeric_data.corr()
print("\nКорреляционная матрица экономических показателей:")
print(correlation_matrix)

# Построение тепловой карты корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица экономических показателей')
plt.show()

# 5. Визуализация трендов по странам
countries = df['Country'].unique()
fig, axs = plt.subplots(len(countries), 2, figsize=(14, 6 * len(countries)))

# Если одна страна, преобразуем оси в одномерный массив
if len(countries) == 1:
    axs = [axs]

# Строим графики для каждой страны
for i, country in enumerate(countries):
    country_data = df[df['Country'] == country]

    aggregated_country_data = country_data.groupby('Year').agg({
        'Экономический рост': 'mean',
        'Уровень безработицы': 'mean'
    }).reset_index()

    # График для "Экономический рост"
    axs[i][0].plot(aggregated_country_data['Year'], aggregated_country_data['Экономический рост'], marker='o')
    axs[i][0].set_title(f'Экономический рост по годам ({country})')
    axs[i][0].set_xlabel('Год')
    axs[i][0].set_ylabel('Экономический рост')

    # График для "Уровень безработицы"
    axs[i][1].plot(aggregated_country_data['Year'], aggregated_country_data['Уровень безработицы'], marker='o',
                   color='red')
    axs[i][1].set_title(f'Уровень безработицы по годам ({country})')
    axs[i][1].set_xlabel('Год')
    axs[i][1].set_ylabel('Уровень безработицы')

plt.subplots_adjust(left=0.086, bottom=0.063, right=0.96, top=0.956, wspace=0.198, hspace=0.8)
plt.show()
