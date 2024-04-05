# import time

import pandas as pd
import plotly.express as px
# import itertools
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# from catboost import CatBoostClassifier
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as imbpipeline
# from lightgbm import LGBMClassifier
# from sklearn.compose import ColumnTransformer
# from sklearn.dummy import DummyClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import RidgeClassifier
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import OneHotEncoder
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


def tele_churn_app():
    logger.info("UI loop")

    # Configure UI
    icon = "🪁"
    st.set_page_config(page_title=APP_DESCRIPTION, page_icon=icon, layout="wide")

    st.title(icon + " " + APP_DESCRIPTION)

    # Explore Dataset
    st.header("Dataset")

    st.markdown("""
                As an example, we use the Customer Churn Prediction dataset of 4250 records.
                > Kostas Diamantaras. (2020). Customer Churn Prediction 2020. Kaggle. https://kaggle.com/competitions/customer-churn-prediction-2020.
                """)

    churn_stats, csv_tab, engineered_features_tab, correlations_tab = st.tabs(
        ["Churn Stats", "CSV", "Engineered features", "Correlations"]
    )

    # Show csv
    df = _load_dataset()

    with csv_tab:
        st.dataframe(df, hide_index=True)

    # Churn stats
    churn_counts = _churn_counts(df)

    with churn_stats:
        fig = px.pie(churn_counts, values="count", names=churn_counts.index, title="Churn Counts")
        st.plotly_chart(fig, use_container_width=True)

    # Engineer features
    df = _engineer_features(df)

    with engineered_features_tab:
        st.dataframe(df, hide_index=True)

    # Correlations
    correlations = df[df.columns[1:]].corr()["churn"][:].sort_values(ascending=False).to_frame()
    closeness_interval = 0.0001
    correlations["collinearity?"] = (
        (correlations["churn"].shift(-1) - correlations["churn"]).abs() < closeness_interval
    ) | ((correlations["churn"].shift(+1) - correlations["churn"]).abs() < closeness_interval)

    with correlations_tab:
        st.dataframe(correlations)

    # st.header("📊 Features")


@st.cache_data
def _load_dataset():
    df = pd.read_csv(DATASET_CSV_PATH)
    df = df.rename(columns=str.lower)
    return df


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
