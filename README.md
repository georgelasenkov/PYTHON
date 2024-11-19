<h1 align="center">PYTHON</h1>
Здесь мы рассмотрим первичный анализ данных (EDA) и, возможно, коснемся машинного обучения 🦾. В нашем распоряжении датасет по продажам б/у автомобилей на 400 000+ строк, в котором есть информация о производителе, цене, пробеге, цвете, состоянии и т.д. Работать будем в Google Colaboratory.

<h2 align="center">🏁 Поехали! 🏁</h2>

### 0️⃣ Устанавливаем/импортируем библиотеки, загружаем датасет
```python
pip install ydata-profiling 
from ydata_profiling import ProfileReport # Сделает подробный отчет о датасете за нас
import pandas as pd
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


























&nbsp;



&nbsp;



&nbsp;





<div align="center">
  <img height="300" width="450" src="https://media.tenor.com/Dh7CxUiogBMAAAAi/vev-veve.gif"  />
</div>
