import os

import pandas as pd
import streamlit as st
from PIL import Image

from churn_prediction.cmd.web.eda import data_cleaning
from churn_prediction.cmd.web.model_structured import building_model

# from .eda import eda
FILE_PATH = os.path.abspath(os.curdir)

st.set_page_config(
    page_title="Home",
    page_icon=":rocket:",
    layout="wide"
)


def main():
    df = pd.read_csv(FILE_PATH + "/churn_prediction/data/raw/BankChurners.csv")

    image_path = os.path.join(os.getcwd(), 'churn_prediction/cmd/web/assets/pic.png')
    image = Image.open(image_path)

    st.sidebar.image(image, width=100)
    st.sidebar.title("Profile")
    st.sidebar.subheader("Bee Bee W.")
    st.sidebar.write("""
        Transitioning from a 4-year tenure as a Software Engineer to the dynamic world of Data Science,
        I bring with me a robust understanding of software development combined with a newfound passion for data analysis.
        Proficient in Python and well-versed in leveraging Tableau for insightful data visualizations,
        I possess a solid foundation in statistics and machine learning
    """)

    st.sidebar.divider()

    st.sidebar.title("Navigation")
    st.sidebar.subheader("[1. Data Cleaning and EDA](#1-data-cleaning-and-eda)")
    st.sidebar.subheader("[2. Building Model](#2-building-model)")

    st.title("Churn Prediction")
    st.write(
        "A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction")

    """
    FROM the dataset explanation, we don't need these 2 columns below
    1. Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1
    2. Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2
    """
    df = df.drop(
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        axis=1)
    df = df.drop(
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        axis=1)

    st.divider()
    data_cleaning(df)

    st.divider()
    building_model()


if __name__ == "__main__":
    main()
