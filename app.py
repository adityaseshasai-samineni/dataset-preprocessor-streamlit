import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os

def home_page():
    if st.button("Data Description"):
        st.session_state.page = "data_description"
    if st.button("Handle Null values"):
        st.session_state.page = "Hand_null_val"
    if st.button("Encode Data"):
        st.session_state.page = "encode_data"
    if st.button("Feature Scaling"):
        st.session_state.page = "feature_scale"
    if st.button("Download the dataset"):
        st.session_state.page = "download"

def data_description_page():
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Encode Data"):
            st.session_state.page = "encode_data"
        if st.button("Feature Scaling"):
            st.session_state.page = "feature_scale"
        if st.button("Download the dataset"):
            st.session_state.page = "download"
            
    st.header("Data Description Page")
    if st.button("Describe a column"):
        st.session_state.page = "column_form"
    if st.button("Dataset's Properties"):
        st.write(st.session_state.df.describe(include='all'))
        buf = StringIO()
        st.session_state.df.info(buf=buf)
        st.text(buf.getvalue())
    if st.button("show DataSet"):
        st.write(st.session_state.df)
    if st.button("Back"):
        st.session_state.page = "home"

def column_describe_page():
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Encode Data"):
            st.session_state.page = "encode_data"
        if st.button("Feature Scaling"):
            st.session_state.page = "feature_scale"
        if st.button("Download the dataset"):
            st.session_state.page = "download"

    with st.form(key='column_form'):
            st.header('Columns')
            st.write(st.session_state.df.columns.to_list())
            col_name = st.text_input("Enter Column Name")
            submit_button = st.form_submit_button("Enter")
            if submit_button:
                if col_name in st.session_state.df.columns.to_list():
                    st.write(st.session_state.df[col_name].describe())
                else:
                    st.write(f"The column '{col_name}' does not exist in the DataFrame.")
            else:
                st.write("The 'Describe a column' button hasn't been clicked.")
    if st.button("Back"):
        st.session_state.page = "data_description"

def input():
    st.title("Dataset Preprocessor")
    st.write("Welcome to the Dataset Preprocessor app! This tool helps you preprocess your datasets.")
    
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
         # Buttons for different actions
        if st.button("continue"):
            st.session_state.page = "home"

def null_val_handle_page():
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Encode Data"):
            st.session_state.page = "encode_data"
        if st.button("Feature Scaling"):
            st.session_state.page = "feature_scale"
        if st.button("Download the dataset"):
            st.session_state.page = "download"

    st.header("Handle NULL values")
    if st.button("Show NULL values"):
        st.write(st.session_state.df.isnull().sum()) 
    if st.button("Remove columns"):
        st.session_state.page = "remove_col"
    if st.button("Fill NULL values"):
        st.session_state.page = "fill_null_val"
    if st.button("show DataSet"):
        st.write(st.session_state.df)
    if st.button("Back"):
        st.session_state.page = "home"

def remove_col_page():
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Encode Data"):
            st.session_state.page = "encode_data"
        if st.button("Feature Scaling"):
            st.session_state.page = "feature_scale"
        if st.button("Download the dataset"):
            st.session_state.page = "download"

    with st.form(key='remove_col'):
            st.header('Remove Columns')
            st.write(st.session_state.df.columns.to_list())
            col_name = st.text_input("Enter Column Name")
            submit_button = st.form_submit_button("Enter")
            if submit_button:
                if col_name in st.session_state.df.columns.to_list():
                    st.session_state.df = st.session_state.df.drop([col_name], axis=1)
                    st.write(f"The column '{col_name}' Dropped")
                else:
                    st.write(f"The column '{col_name}' does not exist in the DataFrame.")
            else:
                st.write("The 'Describe a column' button hasn't been clicked.")
    if st.button("Back"):
        st.session_state.page = "Hand_null_val"

def fill_null_val():
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Encode Data"):
            st.session_state.page = "encode_data"
        if st.button("Feature Scaling"):
            st.session_state.page = "feature_scale"
        if st.button("Download the dataset"):
            st.session_state.page = "download"

    with st.form(key='fill_null_val'):
            st.header('Fill NULL values')
            st.write(st.session_state.df.columns.to_list())
            col_name = st.text_input("Enter Column Name")
            submit_button = st.form_submit_button("Enter")
            if submit_button:
                if col_name in st.session_state.df.columns.to_list():
                    if st.form_submit_button("Mean"):
                        st.session_state.df = st.session_state.df[col_name].fillna(st.session_state.df[col_name].mean(), inplace=True)
                    if st.form_submit_button("Median"):
                        st.session_state.df = st.session_state.df[col_name].fillna(st.session_state.df[col_name].median(), inplace=True)
                    if st.form_submit_button('Mode'):
                        st.session_state.df = st.session_state.df[col_name].fillna(st.session_state.df[col_name].mode(), inplace=True)
                    st.write(f"The column '{col_name}' Filled")
                else:
                    st.write(f"The column '{col_name}' does not exist in the DataFrame.")
            else:
                st.write("The 'Describe a column' button hasn't been clicked.")
    if st.button("Back"):
        st.session_state.page = "Hand_null_val"

def encode_data():
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Encode Data"):
            st.session_state.page = "encode_data"
        if st.button("Feature Scaling"):
            st.session_state.page = "feature_scale"
        if st.button("Download the dataset"):
            st.session_state.page = "download"

    if st.button("Show Categorical Columns"):
        categorical_columns = st.session_state.df.select_dtypes(include=['object'])
        unique_counts = {col: st.session_state.df[col].nunique() for col in categorical_columns}
        unique_counts_df = pd.DataFrame(list(unique_counts.items()), columns=['Column', 'Unique Count'])
        st.table(unique_counts_df)
    
    categorical_columns = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
    selected_column = st.selectbox("Select a column to encode", categorical_columns)

    if st.button("Encode Categorical Columns"):
        st.session_state.df = pd.get_dummies(st.session_state.df, columns=[selected_column])
        st.write(st.session_state.df)
    
    if st.button("show DataSet"):
        st.write(st.session_state.df)

    if st.button("Back"):
        st.session_state.page = "home"

def feature_scale():
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Encode Data"):
            st.session_state.page = "encode_data"
        if st.button("Feature Scaling"):
            st.session_state.page = "feature_scale"
        if st.button("Download the dataset"):
            st.session_state.page = "download"

    numeric_columns = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if "normalize_scaler" not in st.session_state:
        st.session_state.normalize_scaler = False
    if "standardize_scaler" not in st.session_state:
        st.session_state.standardize_scaler = False

    if st.button("Normalization (MinMax Scaler)"):
        st.session_state.normalize_scaler = True

    if st.session_state.normalize_scaler:
        if st.button("Normalize whole dataset"):
            scaler = MinMaxScaler()
            st.session_state.df[numeric_columns] = scaler.fit_transform(st.session_state.df[numeric_columns])
            st.write(st.session_state.df)

        
        st.session_state.selected_column = st.selectbox("Select a column to normalize", numeric_columns)

        if st.button("Normalize a column"):
            scaler = MinMaxScaler()
            st.session_state.df[st.session_state.selected_column] = scaler.fit_transform(st.session_state.df[[st.session_state.selected_column]])
            st.write(st.session_state.df)

    if st.button("Standardization (Standard Scaler)"):
        st.session_state.normalize_scaler = False
        st.session_state.standardize_scaler = True
    
    if st.session_state.standardize_scaler:
        if st.button("Standardize whole dataset"):
            scaler = StandardScaler()
            st.session_state.df[numeric_columns] = scaler.fit_transform(st.session_state.df[numeric_columns])
            st.write(st.session_state.df)

        st.session_state.selected_column = st.selectbox("Select a column to standardize", numeric_columns)

        if st.button("Standardize a column"):
            scaler = StandardScaler()
            st.session_state.df[st.session_state.selected_column] = scaler.fit_transform(st.session_state.df[[st.session_state.selected_column]])
            st.write(st.session_state.df)

    if st.button("show DataSet"):
        st.write(st.session_state.df)
    
    if st.button("Back"):
        st.session_state.normalize_scaler = False
        st.session_state.page = "home"


def download():
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Encode Data"):
            st.session_state.page = "encode_data"
        if st.button("Feature Scaling"):
            st.session_state.page = "feature_scale"
        if st.button("Download the dataset"):
            st.session_state.page = "download"
    
    st.header("Download the preprocessed dataset")
    file_name = st.text_input("Enter the file name", "preprocessed_data.csv")

    if st.button("Save"):
        st.session_state.df.to_csv(file_name, index=False)
        st.markdown(f"File saved as {file_name} in the current directory: {os.getcwd()}")
    if st.button("Back"):
        st.session_state.page = "home"

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "input"
    # Display the selected page
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "data_description":
        data_description_page()
    elif st.session_state.page == "column_form":
        column_describe_page()
    elif st.session_state.page == "main":
        main()
    elif st.session_state.page == "input":
        input()
    elif st.session_state.page == "Hand_null_val":
        null_val_handle_page()
    elif st.session_state.page == "remove_col":
        remove_col_page()
    elif st.session_state.page == "fill_null_val":
        fill_null_val()
    elif st.session_state.page == "encode_data":
        encode_data()
    elif st.session_state.page == "feature_scale":
        feature_scale()
    elif st.session_state.page == "download":
        download()

if __name__ == "__main__":
    main()