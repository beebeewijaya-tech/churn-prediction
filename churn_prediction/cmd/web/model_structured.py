import os

import pandas as pd
import plotly.express as px
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FILE_PATH = os.path.abspath(os.curdir)


def building_model():
    st.header("2. Building Model")
    st.write("On this part, we will build a model using the data we already clean and structured")
    st.write("""
        Rules that we are going to do for our model building
        1. Load data that already been cleaned
        2. Handle Imbalance Data
        3. Split train and test data
        4. Feature scaling
        5. Building model
        6. Predict model
        7. Assessing the result
    """)

    df = load_data()
    X_train, X_test, y_train, y_test = splitting_train_test(df)
    X_train, y_train = handle_imbalance_data(X_train, y_train)

    X_train, X_test, sc = feature_scaling(X_train, X_test)

    rfc = build_model(X_train, y_train)
    y_pred = predict_model(rfc, X_test)

    assessing_result(y_test, y_pred)


def handle_imbalance_data(X_train, y_train):
    st.subheader("Handling Imbalance Data")
    st.write("""
        To understand imbalance data, we need to make a bar graph on our dependent variables.
        To see how much different between both classes.

        For now, we will need to extract the dependent variable and show the bar count.
    """)

    fig = px.histogram(y_train, x="Attrition_Flag", color="Attrition_Flag",
                       color_discrete_sequence=["#F9C901", "#985b10"])
    st.plotly_chart(fig, use_container_width=True)

    st.write("We can see that the `Existing Customer` or 0 class has more sample than the `Churn Customer` or class 1.")
    st.write("To handle this, we need to do resampling with oversampling method to achieve same count of data.")
    st.write("Remember that oversampling only need to be implemented on train data, to avoid leakage to test data")

    st.write("We will using `SMOTE()` function to achieve oversampling")
    sm = SMOTE(random_state=0)

    X_res, y_res = sm.fit_resample(X_train, y_train)
    st.write("""
        ```
        sm = SMOTE(random_state=0)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        ```
    """)

    fig = px.histogram(y_res, x="Attrition_Flag", color="Attrition_Flag",
                       color_discrete_sequence=["#F9C901", "#985b10"])
    st.plotly_chart(fig, use_container_width=True)

    st.write("Now we can see that our training data already has the same weight on both class.")

    st.divider()

    return X_res, y_res


def assessing_result(y_test, y_pred):
    st.subheader("Result Assessment")
    st.write("""
        We already getting result from the prediction, it's time to assess the result before we goes to improving the model.
        
        Couple ways to do it.
        1. Confusion matrix
        2. F1-Score
        3. Precision
        
        We want increase the performance of the true positive, because we want to prevent churn happened.
        We want to prevent reduce false negative value ( we don't want if the churn customer predicted as non-churn )
    """)

    st.write("Let's start with confusion matrix")
    cfm = confusion_matrix(y_test, y_pred)
    st.write(cfm)
    st.write("""
        We can say that our confusion matrix is not that good to predict the class 1 comparing to class 0.
        We almost get half of the class 1 as a false negative.
    """)

    pre = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"""
        Prediction score: {pre:.2f}
        
        F1 score: {f1:.2f}
        
        
        We can see that the score is actually not that bad for a Base Model, we can tuning it up in the next iteration.
    """)


def predict_model(rfc, X_test):
    st.subheader("Predicting the result using RandomForestClassifier()")
    st.write("""
        Now we will predict the X_test using RandomForestClassifier we builded.
        
        Just as simple as using `.predict()` method from the classifier and passing the X_test
    """)

    st.write("`y_pred = rfc.predict(X_test)`")
    y_pred = rfc.predict(X_test)

    st.divider()
    return y_pred


def build_model(X_train, y_train):
    st.subheader("Building the model")
    st.write("""
        We will use the X_train that already scaled to build the model.
        
        Remember that this model is classification model, to predict churn or not churn.
        
        We will using RandomForestClassifier() as base model for this.
        
        for the `n_estimators` we will probably put 20 as the base model and `max_depth` is 6
    """)

    rfc = RandomForestClassifier(n_estimators=20, random_state=0, max_depth=6)

    st.write("We will fit the model `rfc.fit(X_train, y_train)`")
    rfc.fit(X_train, y_train)

    st.write(rfc)
    st.divider()

    return rfc


def feature_scaling(X_train, X_test):
    st.subheader("Feature scale the data")
    st.write("""
        We already make the data into the same weight for both class with same count of data.
        
        Now, the purpose of feature scale is to normalize the value.
        You realize that the value from the datasets we have at the top are varies.
        
        So we need to make them has a same scale value.
    """)

    st.write("We will be using StandardScaler() from sklearn")
    sc = StandardScaler()

    st.write("""
        We will change the value of X_train and X_test only because the values are varies.
        
        `new_xtrain = sc.fit_transform(X_train)`
        
        `new_xtest = sc.transform(X_test)`
        
        `fit_transform` will return the sc to use a mean and normal value from X_train
        
        `transform` will use the value from current sc mean and normal that been generated from X_train
    """)
    new_xtrain = sc.fit_transform(X_train)
    new_xtest = sc.transform(X_test)

    st.write("`new_xtrain[:10]`")
    st.write(new_xtrain[:10])

    st.divider()
    return new_xtrain, new_xtest, sc


def splitting_train_test(df):
    st.subheader("Splitting train and test data")
    st.write("""
        The purpose of splitting is to able verifying the model performance later.
        
        We will need to separate X and y first.
        
        X is independent variables or features
        
        y is dependent variable or target value
        
        
        X = the rest of column, minus Attrition_Flag
        
        y = Attrition_Flag
    """)

    X = df[["Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
            "Total_Revolving_Bal", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]].values
    y = df[["Attrition_Flag"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    st.write("`X_train[:10]`")
    st.write(X_train[:10])

    st.write("`y_train[:10]`")
    st.write(y_train[:10])

    st.divider()
    return X_train, X_test, y_train, y_test


def load_data():
    st.subheader("Let's load the data we already cleaned")
    st.write("`data/processed/churn_cleaned_01.csv`")
    df = pd.read_csv(FILE_PATH + "/churn_prediction/data/processed/churn_cleaned_01.csv")
    st.write(df.head())
    st.divider()
    return df
