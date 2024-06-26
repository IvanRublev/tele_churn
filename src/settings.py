import os
import toml

"""Settings for the application.

Some of them are loaded from environment variables, some hardcoded, and others loaded from the pyproject.toml .
"""

with open("pyproject.toml", "r") as file:
    _pyproject_content = toml.load(file)


# Loaded from environment variables

PORT: int = int(os.environ["STREAMLIT_SERVER_PORT"])

# Hardcoded

DATASET_CSV_PATH: str = "dataset/customer_churn_prediction_2020_train.csv"
INTEGER_GROUPING_SYMBOL = ","
INTEGER_FORMAT = f"{{:{INTEGER_GROUPING_SYMBOL}d}}"
PERCENTAGE_FORMAT = "{:.2f}%"
PROCESSING_TIME_FORMAT = "{:.2f}"
MODEL_ACCURACY_FORMAT = "{:.3f}"

# From pyproject.toml

APP_NAME = _pyproject_content["tool"]["poetry"]["name"]
APP_DESCRIPTION = _pyproject_content["tool"]["poetry"]["description"]
APP_VERSION = _pyproject_content["tool"]["poetry"]["version"]
