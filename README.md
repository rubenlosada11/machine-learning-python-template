# WeLoveReviews — Análisis de sentimiento en reseñas de clientes

Clasifica las 500 reseñas escritas de **Harbor House Café** como positivas, neutrales o negativas con un modelo preentrenado de Hugging Face y comprueba si el tono del texto es coherente con su valoración media de **4,5 / 5 estrellas**. No se entrena ningún modelo.

**Resultado principal:** 82,4 % positivas · 8,2 % neutrales · 9,4 % negativas. El texto respalda la valoración de 4,5, pero el porcentaje de negativas está sobreestimado: 14 de las 16 reseñas de 4–5 ★ que el modelo marca como negativas son elogios mal clasificados. El análisis completo está en [`src/explore.ipynb`](src/explore.ipynb).

## Estructura relevante

| Ruta | Contenido |
|---|---|
| `src/explore.ipynb` | **Documento principal**: EDA, plan, inferencia, resultados, falsos negativos, validación manual, conclusiones y addendum con un segundo modelo. Se entrega ejecutado. |
| `src/app.py` | Script de producción: lee el CSV, clasifica todas las reseñas y escribe el CSV enriquecido. |
| `tests/test_app.py` | Tests unitarios de `app.py` con un modelo simulado (no descargan nada). |
| `data/raw/reviews.csv` | Entrada: `review_id`, `rating` (1–5), `review_text`. No se modifica. |
| `data/processed/reviews_with_sentiment.csv` | Salida: columnas originales + `predicted_stars`, `sentiment`, `model_label`, `model_score`. |
| `requirements.txt` | Dependencias con versiones fijadas (PyTorch solo CPU). |

## Instalación

Requisitos: **Python 3.13** (la versión del devcontainer, con la que se ha probado) y unos 2 GB de disco para dependencias y modelo.

```bash
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (Git Bash):     source .venv/Scripts/activate
# Linux / macOS:          source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` añade el índice `https://download.pytorch.org/whl/cpu` para instalar PyTorch **solo CPU** y evitar la versión CUDA, de ~2 GB, en Linux/Codespaces. No hace falta GPU.

## Ejecución

Todos los comandos se ejecutan desde la raíz del repositorio y con el entorno activado.

```bash
# 1. Script de producción -> data/processed/reviews_with_sentiment.csv
python src/app.py

# 2. Notebook de análisis (re-ejecuta todas las celdas y guarda los outputs)
jupyter nbconvert --to notebook --execute --inplace src/explore.ipynb

# 3. Tests
python -m pytest tests
```

El notebook también se puede abrir en VS Code o Jupyter seleccionando el kernel del `.venv`. Su anexo final comprueba que sus predicciones coinciden con el CSV generado por `app.py`, así que conviene ejecutar antes el script.

`app.py` devuelve el código de salida `0` si todo va bien, `1` si hay un error en los datos de entrada (archivo inexistente o vacío, columnas ausentes, rating fuera de 1–5, `review_id` duplicado) y `2` si falla la carga del modelo o la inferencia. Las reseñas con texto vacío se conservan en la salida, con las columnas de predicción vacías.

## Modelo y clasificación

- Modelo: [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) (BERT multilingüe, licencia MIT), revisión fijada en el commit `8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d`. Predice de 1 a 5 estrellas.
- Bandas de sentimiento: **1–2 ★ → Negativo · 3 ★ → Neutral · 4–5 ★ → Positivo**.
- `model_score` es la probabilidad *softmax* de la clase elegida. **No** es una probabilidad calibrada de acierto.

### Descarga y caché del modelo

En la primera ejecución, `transformers` descarga los pesos (~670 MB) a la caché de Hugging Face (`~/.cache/huggingface`; en Windows, `C:\Users\<usuario>\.cache\huggingface`). Las ejecuciones siguientes la reutilizan, y con la caché ya llena funciona sin conexión (`HF_HUB_OFFLINE=1`). Los pesos **no** se guardan en el repositorio (`.gitignore` excluye `*.safetensors`, `*.bin`, `*.pt`…). Para usar otra carpeta, define `HF_HOME`. En Windows puede aparecer un aviso sobre enlaces simbólicos en la caché: es inofensivo.

## Limitaciones conocidas

- **Sentimiento ≠ rating.** El rating es lo que el cliente puntuó y el sentimiento es el tono que el modelo detecta en el texto. Pueden diferir legítimamente: una reseña de 5 ★ puede mencionar una queja. Ninguno de los dos es «la verdad» del otro.
- **Desajuste de dominio.** El modelo se ajustó con reseñas de productos y aquí se aplica a reseñas de un servicio. Falla de forma sistemática con frases de hostelería como *«Every dish was bursting with flavor»* o *«accommodate my allergy»*, a las que da 1 estrella aunque sean elogios. Que la causa sea el dominio de entrenamiento es una hipótesis.
- **Dataset plantillado.** Las reseñas combinan solo 69 frases distintas, así que cada error se repite muchas veces. Los resultados no se pueden extrapolar sin más a reseñas reales.
- **Validación limitada.** La revisión manual (18 reseñas de control y 16 discordancias) la hizo el analista que construyó el notebook. No es una evaluación formal sobre un conjunto etiquetado de forma independiente.
- El modelo alternativo del addendum (`tabularisai/multilingual-sentiment-analysis`) tiene licencia **CC BY-NC 4.0 (no comercial)** y solo se usa para comparar.

---

## Template notes (original 4Geeks boilerplate)

This boilerplate is designed to kickstart data science projects by providing a basic setup for database connections, data processing, and machine learning model development. It includes a structured folder organization for your datasets and a set of pre-defined Python packages necessary for most data science tasks.

## Structure

The project is organized as follows:

- **`src/app.py`** → Main Python script where your project will run.
- **`src/explore.ipynb`** → Notebook for exploration and testing. Once exploration is complete, migrate the clean code to `app.py`.
- **`src/utils.py`** → Auxiliary functions, such as database connection.
- **`requirements.txt`** → List of required Python packages.
- **`models/`** → Will contain your SQLAlchemy model classes.
- **`data/`** → Stores datasets at different stages:
  - **`data/raw/`** → Raw data.
  - **`data/interim/`** → Temporarily transformed data.
  - **`data/processed/`** → Data ready for analysis.


## ⚡ Initial Setup in Codespaces (Recommended)

No manual setup is required, as **Codespaces is automatically configured** with the predefined files created by the academy for you. Just follow these steps:

1. **Wait for the environment to configure automatically**.
   - All necessary packages and the database will install themselves.
   - The automatically created `username` and `db_name` are in the **`.env`** file at the root of the project.
2. **Once Codespaces is ready, you can start working immediately**.


## 💻 Local Setup (Only if you can't use Codespaces)

**Prerequisites**

Make sure you have Python 3.11+ installed on your machine. You will also need pip to install the Python packages.

**Installation**

Clone the project repository to your local machine.

Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

**Create a database (if necessary)**

Create a new database within the Postgres engine by customizing and executing the following command:

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER my_user WITH PASSWORD 'my_password'; 
    CREATE DATABASE my_database OWNER my_user; 
END \$\$;"
```
Connect to the Postgres engine to use your database, manipulate tables, and data:

```bash
$ psql -U my_user -d my_database
```

Once inside PSQL, you can create tables, run queries, insert, update, or delete data, and much more!

**Environment Variables**

Create a .env file in the root directory of the project to store your environment variables, such as your database connection string:

```makefile
DATABASE_URL="postgresql://<USER>:<PASSWORD>@<HOST>:<PORT>/<DB_NAME>"

#example
DATABASE_URL="postgresql://my_user:my_password@localhost:5432/my_database"
```

## Running the Application

To run the application, execute the app.py script from the root directory of the project:

```bash
python src/app.py
```

## Adding Models

To add SQLAlchemy model classes, create new Python script files within the models/ directory. These classes should be defined according to your database schema.

Example model definition (`models/example_model.py`):

```py
from sqlalchemy.orm import declarative_base
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Working with Data

You can place your raw datasets in the data/raw directory, intermediate datasets in data/interim, and processed datasets ready for analysis in data/processed.

To process data, you can modify the app.py script to include your data processing steps, using pandas for data manipulation and analysis.

## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.