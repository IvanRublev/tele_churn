# import time

import pandas as pd
import plotly.express as px
# import itertools
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline

# from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer

# from sklearn.dummy import DummyClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer

# from sklearn.linear_model import RidgeClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# from sklearn.naive_bayes import BernoulliNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import ExtraTreeClassifier
import streamlit as st
# from xgboost import XGBClassifier

from src.logger import logger
from src.settings import APP_DESCRIPTION
from src.settings import DATASET_CSV_PATH
from src.settings import INTEGER_GROUPING_SYMBOL
from src.settings import INTEGER_FORMAT
from src.settings import PERCENTAGE_FORMAT


def tele_churn_app():
    logger.info("UI loop")

    # Configure UI
    icon = "🪁"
    st.set_page_config(page_title=APP_DESCRIPTION, page_icon=icon, layout="wide")

    st.title(icon + " " + APP_DESCRIPTION)

    # Explore Dataset
    st.header("📚 Dataset")

    st.markdown("""
                As an example, we use the Customer Churn Prediction dataset of 4250 records.
                > Kostas Diamantaras. (2020). Customer Churn Prediction 2020. Kaggle. https://kaggle.com/competitions/customer-churn-prediction-2020.
                """)

    churn_stats, csv_tab, engineered_features_tab, correlations_tab, x_tab = st.tabs(
        ["Churn Stats", "CSV", "Engineered features", "Correlations", "X features for model training"]
    )

    # Show csv
    df = _sort_columns(_load_dataset())

    with csv_tab:
        st.dataframe(df, hide_index=True)

    # Churn stats
    churn_counts = _churn_counts(df)

    with churn_stats:
        fig = px.pie(churn_counts, values="count", names=churn_counts.index, title="Churn Counts")
        st.plotly_chart(fig, use_container_width=True)

    # Engineer features
    df = _sort_columns(_engineer_features(df))

    with engineered_features_tab:
        st.dataframe(df, hide_index=True)

    # Correlations
    correlations = _correlations(df, closeness_interval=0.0001)

    with correlations_tab:
        st.dataframe(correlations)

    # Drop collinear features and target variable
    y = df["churn"]

    dropped_columns = [
        "churn",
        "international_plan",
        "total_day_charge",
        "total_eve_charge",
        "total_intl_charge",
        "total_night_charge",
    ]

    X = _drop_columns(df, dropped_columns)

    with x_tab:
        dropped_columns.sort()
        dropped_columns_str = ", ".join(map(lambda x: f"`{x}`", dropped_columns))
        st.write(f"The following columns were dropped: {dropped_columns_str}")
        st.dataframe(X, hide_index=False)

    st.header("⚽ Model Training")

    # Split dataset
    ttf, train_Xy, test_Xy, X_train, X_test, y_train, y_test = _split_dataset(X, y)

    split_stats, train_col, test_col = st.columns(3)

    with split_stats:
        ttf["Percentage"] = ttf["Percentage"].apply(lambda x: PERCENTAGE_FORMAT.format(x))
        fig = px.bar(
            ttf,
            x="Dataset",
            y="Count",
            color="Churn",
            title="Split to Train and Test Datasets",
            barmode="stack",
            hover_data=["Dataset", "Churn", "Percentage"],
        )
        fig.update_yaxes(tickformat=INTEGER_GROUPING_SYMBOL)
        st.plotly_chart(fig, use_container_width=True)

    with train_col:
        train_len_str = INTEGER_FORMAT.format(len(train_Xy))
        st.subheader(f"Train Set ({train_len_str} records)")
        st.dataframe(train_Xy, hide_index=False)

    with test_col:
        test_len_str = INTEGER_FORMAT.format(len(test_Xy))
        st.subheader(f"Test Set ({test_len_str} records)")
        st.dataframe(test_Xy, hide_index=False)

    st.markdown("""
            In the pipeline we use the following to preprocess the input dataset:
            * [SimpleImputer][SimpleImputer] to fill missing values
            * [OneHotEncoder][OneHotEncoder] to encode categorical features with automatic categories detection
            * [SMOTE][SMOTE] to balance the dataset
            * [MinMaxScaler][MinMaxScaler] to normalize numerical features in range `0..1`
                
            [SimpleImputer]: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
            [OneHotEncoder]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
            [SMOTE]: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
            [MinMaxScaler]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
            """)
    # _normalized_input_pipeline()


@st.cache_data
def _load_dataset():
    df = pd.read_csv(DATASET_CSV_PATH)
    df = df.rename(columns=str.lower)
    return df


@st.cache_data
def _sort_columns(df):
    return df.sort_index(axis=1)


@st.cache_data
def _churn_counts(df):
    return df["churn"].value_counts()


@st.cache_data
def _engineer_features(df):
    pd.set_option("future.no_silent_downcasting", True)
    df["churn"] = df["churn"].replace(("yes", "no"), (1, 0))
    df["international_plan"] = df["international_plan"].replace(("yes", "no"), (1, 0))
    df["voice_mail_plan"] = df["voice_mail_plan"].replace(("yes", "no"), (1, 0))
    df["charge_rate_day"] = _call_charge_rate(df, "total_day_minutes", "total_day_charge")
    df["charge_rate_night"] = _call_charge_rate(df, "total_night_minutes", "total_night_charge")
    df["charge_rate_intl"] = _call_charge_rate(df, "total_intl_minutes", "total_intl_charge")
    df["charge_rate_eve"] = _call_charge_rate(df, "total_eve_minutes", "total_eve_charge")
    df["mean_encoded_state"] = _mean_encode(df, "state", "churn")
    df["mean_encoded_international_plan"] = _mean_encode(df, "international_plan", "churn")
    df["mean_encoded_voice_mail_plan"] = _mean_encode(df, "voice_mail_plan", "churn")
    df = pd.get_dummies(df, columns=["state", "area_code"])
    return df


def _call_charge_rate(df, minutes_column, charges_column):
    return df[charges_column] / df[minutes_column]


def _mean_encode(df, group, target):
    """Group a Pandas DataFrame via a given column and return
    the mean of the target variable for that grouping.
    Args:
        :param df: Pandas DataFrame.
        :param group: Column to group by.
        :param target: Target variable column.
    Returns:
        Mean for the target variable across the group.
    Example:
        df['sector_mean_encoded'] = _mean_encode(df, 'sector', 'converted')
    """

    mean_encoded = df.groupby(group)[target].mean()
    return df[group].map(mean_encoded)


@st.cache_data
def _correlations(df, closeness_interval):
    closeness_interval = 0.0001
    correlations = df[df.columns[1:]].corr()["churn"][:].sort_values(ascending=False).to_frame()
    correlations["collinearity?"] = (
        (correlations["churn"].shift(-1) - correlations["churn"]).abs() < closeness_interval
    ) | ((correlations["churn"].shift(+1) - correlations["churn"]).abs() < closeness_interval)
    return correlations


@st.cache_data
def _drop_columns(df, columns):
    return df.drop(columns=columns)


@st.cache_data
def _split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    train_Xy = X_train.copy()
    train_Xy["churn"] = y_train

    test_Xy = X_test.copy()
    test_Xy["churn"] = y_test

    train_churn_counts = train_Xy["churn"].value_counts()
    test_churn_counts = test_Xy["churn"].value_counts()
    train_churn_percentages = train_churn_counts / train_churn_counts.sum() * 100
    test_churn_percentages = test_churn_counts / test_churn_counts.sum() * 100

    data = {
        "Dataset": ["Train", "Train", "Test", "Test"],
        "Churn": ["0", "1", "0", "1"],
        "Count": [train_churn_counts[0], train_churn_counts[1], test_churn_counts[0], test_churn_counts[1]],
        "Percentage": [
            train_churn_percentages[0],
            train_churn_percentages[1],
            test_churn_percentages[0],
            test_churn_percentages[1],
        ],
    }
    ttf = pd.DataFrame(data)

    return ttf, train_Xy, test_Xy, X_train, X_test, y_train, y_test


@st.cache_data
def _normalized_input_pipeline(X, model):
    """Return a pipeline to normalize numerical and categorical data bundled together with a given model.

    Args:
        X (object): X_train data.
        model (object): scikit-learn model object, i.e. XGBClassifier

    Returns:
        Pipeline (object): Pipeline steps.
    """

    numeric_columns = list(X.select_dtypes(exclude=["object"]).columns.values.tolist())
    categorical_columns = list(X.select_dtypes(include=["object"]).columns.values.tolist())
    numeric_pipeline = SimpleImputer(strategy="constant")
    categorical_pipeline = OneHotEncoder(handle_unknown="error")  # ignore

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="passthrough",
    )

    bundled_pipeline = imbpipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("scaler", MinMaxScaler()),
            ("model", model),
        ]
    )

    return bundled_pipeline
