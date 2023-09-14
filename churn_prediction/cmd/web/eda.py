import os

import plotly.graph_objects as go
import streamlit as st

FILE_PATH = os.path.abspath(os.curdir)


def data_cleaning(df):
    st.header("1. Data Cleaning and EDA")
    st.write("We will analyzing and doing data cleansing on the data")
    st.write("""
        Rules that we are going to do for our data cleansing and EDA
        1. Columns explanation
        2. Brief Explanation on what we have on the data, the descriptive statistics
        3. Change column type to numbers
        4. Remove duplicate data
        5. Handle null value
        6. Handle Missing data
        7. Check correlation between column
        8. Remove uncorrelated column
        9. Save cleaned datasets
    """)

    st.divider()

    column_explain(df)
    brief_explanation(df)
    change_col_type_to_numbers(df)
    remove_duplicate(df)
    handle_null_value(df)
    df = check_correlation(df)
    save_data(df)


def save_data(df):
    st.write("We will save the data to processed the model later")
    st.write("`data/processed/churn_cleaned_01.csv`")
    df.to_csv(FILE_PATH + "/churn_prediction/data/processed/churn_cleaned_01.csv", index=False)


def change_col_type_to_numbers(df):
    st.subheader("Change column type to numbers")
    st.write("""
           We have a lot of columns that is categorical but important value, to do EDA properly, we need to set it up to
           at least a numeric value like 0, 1, 2, 3, 4 etc

           1. Attrition_Flag, Existing ( 0 ) & Attrited ( 1 )
           2. Gender, M ( 0 ) & F ( 1 )
           3. Education_Level, Unknown ( 0 ) & High School ( 1 ) & Graduate ( 2 ) & Uneducated ( 3 ) & College ( 4 ) & Post-Graduate ( 5 ) & Doctorate ( 6 )
           4. Marital_Status, Unknown ( 0 ) & Single ( 1 ) & Married ( 2 ) & Divorced ( 3 )
           5. Income_Category, Unknown (0) & < 40k ( 1 ), 40k - 60k ( 2 ), 60k - 80k ( 3 ), 80k - 120k ( 4 ), > 120k ( 5 )
           6. Card_Category, Blue ( 0 ), Silver ( 1 ), Gold ( 2 ), Platinum ( 3 )
       """)

    attrition_flag_map = {'Existing Customer': 0, 'Attrited Customer': 1}
    df["Attrition_Flag"] = df["Attrition_Flag"].map(attrition_flag_map)

    gender_map = {'M': 0, 'F': 1}
    df["Gender"] = df["Gender"].map(gender_map)

    education_map = {'Unknown': 0, 'High School': 1, 'Graduate': 2, 'Uneducated': 3, 'College': 4, 'Post-Graduate': 5,
                     'Doctorate': 6}
    df["Education_Level"] = df["Education_Level"].map(education_map)

    marital_map = {'Unknown': 0, 'Single': 1, 'Married': 2, 'Divorced': 3}
    df["Marital_Status"] = df["Marital_Status"].map(marital_map)

    income_map = {'Unknown': 0, 'Less than $40K': 1, '$40K - $60K': 2, '$60K - $80K': 3, '$80K - $120K': 4,
                  '$120K +': 5}
    df["Income_Category"] = df["Income_Category"].map(income_map)

    card_map = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    df["Card_Category"] = df["Card_Category"].map(card_map)

    st.write("Now it would look like this, based on our mapping above")
    st.write(df.head(100))
    st.divider()


def check_correlation(df):
    st.subheader("Analyzing correlation between features")
    st.write("""
        Because we have a lot of columns, let's see which column has highly correlated, we need to remove highly correlated column.
        Remember, not all of the correlated, just the highest one.
        The reason is, highly correlated features, will not give much information. 
        
        We can do it by
        1. Heatmap Plotting
    """)

    corr = df.corr()
    trace = go.Heatmap(
        x=corr.index.values,
        y=corr.columns.values,
        z=corr.values,
        text=corr.values,
        texttemplate='%{text:.2f}',
        colorscale=[[0.0, '#fff6d1'],
                    [0.2, '#ffe373'],
                    [0.4, '#f9c901'],
                    [0.6, '#c28100'],
                    [0.8, '#896800'],
                    [1.0, '#6b4701']]
    )

    fig = go.Figure()
    fig.add_trace(trace)
    fig.update_layout(height=800)

    st.plotly_chart(fig, use_container_width=True, height=800)

    st.write("""
        We can see that our dependent variables has correlated with some of variables greater than 0.15.
        Our variables don't really have big correlation to each others, so we can cut off at 0.15.
    """)

    dependent_var_corr = abs(corr["Attrition_Flag"])
    corr_015 = dependent_var_corr[dependent_var_corr >= 0.15]

    st.table(corr_015)

    st.write("But, you can see on the heatmap that Total_Trans_Amt and Total_Trans_Ct are highly correlated to 0.81")
    st.table(df[["Total_Trans_Amt", "Total_Trans_Ct"]].corr())
    st.write("We need to remove one of this value and we are good to go")

    new_df = df[["Attrition_Flag", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
                 "Total_Revolving_Bal", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]]

    st.write("""
        This would be our new columns after we're only include the correlation one.
        
        `new_df = df[["Attrition_Flag", "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
                 "Total_Revolving_Bal", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]]`
    """)
    st.table(new_df.head())

    st.divider()
    return new_df


def handle_null_value(df):
    st.subheader("Handle null value")
    st.write("Let's analyzing null value data and remove it if exists")
    st.write("Cleaning missing value will really help us on better result later.")
    st.write("`df.isnull().sum()`")
    st.table(df.isnull().sum())
    st.write("We don't have NULL value on the dataset")
    st.write("`df.isna().sum()`")
    st.table(df.isna().sum())
    st.write("We don't have NaN value on the dataset")

    st.write("So, we would assume that we are safe on missing data, and we can continue to handle the next step.")

    st.divider()


def remove_duplicate(df):
    st.subheader("Remove duplicate data")
    st.write("Let's analyzing duplicate data and remove it if exists")
    st.write(df[df.duplicated()])
    st.write("Looks like we don't really have duplicate data, but we can still run `df.drop_duplicates()`")
    df.drop_duplicates()

    st.divider()


def brief_explanation(df):
    st.subheader("Brief understanding on what we have")
    st.write("Before diving into any dataset, it is always a good practice to understand the scope of the data.")

    st.write("Let's print out the 5 top data `df.head()`")
    st.write(df.head())
    st.write("Let's print out the 5 bottom data `df.tail()`")
    st.write(df.tail())

    st.write("Let's print out the description statistics `df.describe()`")
    st.write(df.describe())

    st.divider()


def column_explain(df):
    st.subheader("Columns explanation")
    st.write("Before start anything else, we need to explain the columns")
    st.table(df.columns)
    st.write("""
            All columns doesn't explain anything, we don't know which features is important and what are they used for.
            We must explain the columns on their purposes.

            We have 21 columns right here.

            1. CLIENTNUM: Client number. Unique identifier for the customer holding the account
            2. Attrition_Flag: Internal event (customer activity) variable - if the account is closed then 1 else 0
            3. Customer_Age: Customer's Age in Years
            4. Gender: Customer's Gender M=Male, F=Female
            5. Dependent_count: Number of dependents
            6. Education_Level: Educational Qualification of the account holder (example: high school, college graduate, etc.)
            7. Marital_Status: Married, Single, Divorced, Unknown
            8. Income_Category: Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, > $120K, Unknown)
            9. Card_Category: Type of Card (Blue, Silver, Gold, Platinum)
            10. Months_on_book: Period of relationship with bank
            11. Total_Relationship_Count: Total no. of products held by the customer
            12. Months_Inactive_12_mon: No. of months inactive in the last 12 months
            13. Contacts_Count_12_mon: No. of Contacts in the last 12 months
            14. Credit_Limit: Credit Limit on the Credit Card
            15. Total_Revolving_Bal: Total Revolving Balance on the Credit Card
            16. Avg_Open_To_Buy: Open to Buy Credit Line (Average of last 12 months)
            17. Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)
            18. Total_Trans_Amt: Total Transaction Amount (Last 12 months)
            19. Total_Trans_Ct: Total Transaction Count (Last 12 months)
            20. Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)
            21. Avg_Utilization_Ratio: Average Card Utilization Ratio
        """)

    st.divider()
