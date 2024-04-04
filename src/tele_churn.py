import time

import pandas as pd
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
import streamlit as st
from xgboost import XGBClassifier

from src.logger import logger
from src.settings import APP_DESCRIPTION


def tele_churn_app():
    logger.info("UI loop")

    # Configure UI
    icon = "🪁"
    st.set_page_config(page_title=APP_DESCRIPTION, page_icon=icon, layout="wide")

    st.title(icon + " " + APP_DESCRIPTION)

    st.header("Dataset")

    st.markdown(f"""
                As an example, we use the Customer Churn Prediction dataset of 4250 records.
                > Kostas Diamantaras. (2020). Customer Churn Prediction 2020. Kaggle. https://kaggle.com/competitions/customer-churn-prediction-2020.
                """)
