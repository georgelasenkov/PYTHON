<h1 align="center">PYTHON</h1>
Здесь мы рассмотрим первичный анализ данных (EDA) и, возможно, коснемся машинного обучения 🦾. В нашем распоряжении датасет по продажам б/у автомобилей, в котором есть информация о производителе, цене, пробеге, цвете, состоянии и т.д. Работать будем в Google Colaboratory.

<h2 align="center">🏁 Поехали! 🏁</h2>

### 0️⃣ Устанавливаем/импортируем библиотеки, загружаем датасет
```python
pip install ydata-profiling 
from ydata_profiling import ProfileReport # Сделает подробный отчет о датасете за нас
import pandas as pd
import numpy as np
import plotly.express as px
import kagglehub
path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")
print("Path to dataset files:", path)
  ```
Получаем ссылку на скачивание файла, загружаем его в пандас
```python
url = 'https://www.kaggle.com/api/v1/datasets/download/austinreese/craigslist-carstrucks-data?dataset_version_number=10'
df = pd.read_csv(url, compression='zip') # Автоматическая распаковка архива и чтение csv файла из него
  ```
Датасет загружен и готов

### 1️⃣ Смотрим что внутри и очищаем данные
```python
df.info() # Краткое описание датафрейма
  ```

Результат:

RangeIndex: 426880 entries, 0 to 426879  
Data columns (total 26 columns):
| #   | Column        | Non-Null Count   | Dtype    |
|-----|---------------|------------------|----------|
| 0   | id            | 426880 non-null   | int64    |
| 1   | url           | 426880 non-null   | object   |
| 2   | region        | 426880 non-null   | object   |
| 3   | region_url    | 426880 non-null   | object   |
| 4   | price         | 426880 non-null   | int64    |
| 5   | year          | 425675 non-null   | float64  |
| 6   | manufacturer  | 409234 non-null   | object   |
| 7   | model         | 421603 non-null   | object   |
| 8   | condition     | 252776 non-null   | object   |
| 9   | cylinders     | 249202 non-null   | object   |
| 10  | fuel          | 423867 non-null   | object   |
| 11  | odometer      | 422480 non-null   | float64  |
| 12  | title_status  | 418638 non-null   | object   |
| 13  | transmission  | 424324 non-null   | object   |
| 14  | VIN           | 265838 non-null   | object   |
| 15  | drive         | 296313 non-null   | object   |
| 16  | size          | 120519 non-null   | object   |
| 17  | type          | 334022 non-null   | object   |
| 18  | paint_color   | 296677 non-null   | object   |
| 19  | image_url     | 426812 non-null   | object   |
| 20  | description   | 426810 non-null   | object   |
| 21  | county        | 0 non-null        | float64  |
| 22  | state         | 426880 non-null   | object   |
| 23  | lat           | 420331 non-null   | float64  |
| 24  | long          | 420331 non-null   | float64  |
| 25  | posting_date  | 426812 non-null   | object   |

dtypes: float64(5), int64(2), object(19)  
```python
# Удаляем стобцы с неинтересующими нас данными
df.drop(columns=['id', 'url', 'region', 'region_url', 'VIN', 'size', 'image_url', 'description', 'county', 'state', 'lat', 'long', 'posting_date', 'model', 'cylinders'], inplace=True)
  ```
```python
df.isna().sum()  # Смотрим в скольких ячейках в каждом столбце у нас null значения
  ```

| Column        | Zero Count    |
|---------------|----------------|
| price         | 0              |
| year          | 1205           |
| manufacturer  | 17646          |
| condition     | 174104         |
| fuel          | 3013           |
| odometer      | 4400           |
| title_status  | 8242           |
| transmission  | 2556           |
| drive         | 130567         |
| type          | 92858          |
| paint_color   | 130203         |


```python
df = df.dropna()  # Удаляем строки с null значениями
  ```

На данный момент наш датасет выглядит так:

|  | Price | Year   | Manufacturer      | Model                       | Condition | Fuel | Odometer  | Title Status | Transmission | Drive | Type   | Paint Color |
|--------|-------|--------|-------------------|-----------------------------|-----------|------|-----------|--------------|--------------|-------|--------|-------------|
| 31     | 15000 | 2013.0 | ford              | f-150 xlt                 | excellent | gas  | 128000.0  | clean        | automatic     | rwd   | truck  | black       |
| 32     | 27990 | 2012.0 | gmc               | sierra 2500 hd extended cab | good      | gas  | 68696.0   | clean        | other         | 4wd   | pickup | black       |
| 33     | 34590 | 2016.0 | chevrolet         | silverado 1500 double      | good      | gas  | 29499.0   | clean        | other         | 4wd   | pickup | silver      |
| 34     | 35000 | 2019.0 | toyota           | tacoma                     | excellent | gas  | 43000.0   | clean        | automatic     | 4wd   | truck  | grey        |
| 35     | 29990 | 2016.0 | chevrolet         | colorado extended cab       | good      | gas  | 17302.0   | clean        | other         | 4wd   | pickup | red         |
| ...     | ... | ... | ...         | ...       | ...     | ...  | ...   | ...        | ...         | ...   | ... | ...         |
| 426872 | 32590 | 2020.0 | mercedes-benz     | c-class c 300              | good      | gas  | 19059.0   | clean        | other         | rwd   | sedan  | white       |
| 426873 | 30990 | 2018.0 | mercedes-benz     | glc 300 sport              | good      | gas  | 15080.0   | clean        | automatic     | rwd   | other  | white       |
| 426874 | 33590 | 2018.0 | lexus            | gs 350 sedan 4d            | good      | gas  | 30814.0   | clean        | automatic     | rwd   | sedan  | white       |
| 426876 | 30590 | 2020.0 | volvo            | s60 t5 momentum sedan 4d    | good      | gas  | 12029.0   | clean        | other         | fwd   | sedan  | red         |
| 426878 | 28990 | 2018.0 | lexus            | es 350 sedan 4d            | good      | gas  | 30112.0   | clean        | other         | fwd   | sedan  | silver      |

143333 rows × 11 columns

```python
df.duplicated().sum()  # Поиск дубликатов
  ```
53966 дубликатов

```python
df = df.drop_duplicates()  # Удаляем дубликаты
  ```

```python
# Определим столбцы с числовыми типами переменных
# Используя plotly, построим боксплоты для столбцов с числовыми типами переменных, чтобы найти выбросы
df_num_col = df.select_dtypes(include = [np.number])
for i in df_num_col:
    fig = px.box(df, x = df[i])
    fig.update_traces()
    fig.show()
  ```

price
<h3 align="center"><img src="https://github.com/georgelasenkov/PYTHON/blob/main/px_price.png"></h3>
year
<h3 align="center"><img src="https://github.com/georgelasenkov/PYTHON/blob/main/px_year.png"></h3>
odometer
<h3 align="center"><img src="https://github.com/georgelasenkov/PYTHON/blob/main/px_odometer.png"></h3>

Инсайты:  
- Цены свыше 36000 - выбросы (особенно цены в 1.1111111 млрд и 3.7 млрд 😆). Медианная стоимость - 9500.
- Даты производства ниже 1990 года - выбросы. Самый старый автомобиль - 1900 года.
- Пробег на одометре: нижняя граница ящика с усами - 0, медиана 110000, верхняя граница - 285000. Все что выше - выбросы.

```python
# Удаляю выбросы
s1 = df.shape
clean = df[['price', 'year', 'odometer']]
for i in clean.columns:
    qt1 = df[i].quantile(0.25)
    qt3 = df[i].quantile(0.75)
    iqr =  qt3 - qt1
    lower = qt1-(1.5*iqr)
    upper = qt3+(1.5*iqr)
    min_in = df[df[i]<lower].index
    max_in = df[df[i]>upper].index
    df.drop(min_in, inplace = True)
    df.drop(max_in, inplace = True)
s2 = df.shape
outliers = s1[0] - s2[0]
print("Number of deleted outliers: ", outliers)
  ```
Number of deleted outliers :  9929

```python
# Снова нарисуем ящики для проверки
for i in df_num_col:
    fig = px.box(df, x = df[i])
    fig.update_traces()
    fig.show()
  ```
Результат
<h3 align="center"><img src="https://github.com/georgelasenkov/PYTHON/blob/main/px_outliers_deleted.png"></h3>  

Можем вывести описание для числовых колонок
```python
df.describe()
  ```
| name          |    price |   year |   odometer |
|---------------|----------|--------|------------|
| count         | 79438.0  | 79438.0 | 79438.0    |
| mean          | 11207.647474 | 2009.959062 | 115825.845754 |
| std           | 8310.567213 | 5.831467 | 59060.861023 |
| min           | 0.0      | 1992.0 | 0.0        |
| 25%           | 4995.0   | 2006.0 | 74000.0    |
| 50%           | 8995.0   | 2011.0 | 114456.5   |
| 75%           | 15950.0  | 2014.0 | 155357.25  |
| max           | 36000.0  | 2022.0 | 281794.0   |  

А можем создать отчет с помощью библиотеки ydata-profiling (раньше называлась pandas-profiling). В нем будут все столбцы, которые есть в нашем датафрейме на текущий момент.

```python
# Создаем отчет и выводим на экран
profile = ProfileReport(df, minimal=False, progress_bar=True)
profile
  ```

В разделе "Overview" можно увидет общую информацию о датафрейме и информацию о корреляции между столбцами, если она есть
<h3 align="center"><img src="https://github.com/georgelasenkov/PYTHON/blob/main/profile_overview1.png"></h3>  
Здесь же мы видим что в столбце с ценой есть 3075 записей с ценой автомобиля равной 0, а также несколько значений со странной ценой в 1, 3, 4 и так далее 
<h3 align="center"><img src="https://github.com/georgelasenkov/PYTHON/blob/main/profile_price1.png"></h3>  


&nbsp;

Добавим еще одно условие к нашему датафрейму, установим что цена должна быть не ниже 200.

```python
df = df.loc[df['price'] > 200]
  ```
Снова создадим отчет и посмотрим всю имеющуюся информацию о датафрейме





&nbsp;



&nbsp;



&nbsp;





<div align="center">
  <img height="300" width="450" src="https://media.tenor.com/Dh7CxUiogBMAAAAi/vev-veve.gif"  />
</div>
