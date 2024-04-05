import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os
import base64

def side_bar():
    st.session_state.confirm_reset = False
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
        if st.button("Reset DataFrame"):
            st.session_state.page = "reset"
        if st.button("Work with another dataset"):
            st.session_state.page = "confirm_page"
    
def confirm_page():
    side_bar()
    st.title("Work with another dataset")
    st.write("All the work done on the current dataset will be lost.")
    st.write("Are you sure you want to proceed?")
    if st.button("Yes"):
        st.session_state.page = "input"
        st.experimental_rerun()
    if st.button("No"):
        st.session_state.page = "home"
        st.experimental_rerun()
 
def reset_page():
    side_bar()
    st.title("Reset DataFrame")
    st.write("This action reverses all the operations performed on the DataFrame.")
    st.write("To proceed, enter 'reset' in the text box and click the 'Submit' button.")
    user_input = st.text_input("Enter 'reset' to confirm reset:")
    if st.button("Submit") and user_input == "reset":
        st.session_state.df = st.session_state.original_df.copy()
        st.experimental_rerun()

def home_page():
    if st.button("Data Description"):
        st.session_state.page = "data_description"
    if st.button("Handle Null values"):
        st.session_state.page = "Hand_null_val"
    if st.button("Encode Data"):
        st.session_state.page = "encode_data"
    if st.button("Feature Scaling"):
        st.session_state.page = "feature_scale"
    if st.button("Work with another dataset"):
        st.warning("All the work done on the current dataset will be lost.")
        if st.button("Proceed"):
            st.session_state.page = "input"
        elif st.button("Cancel"):
            st.experimental_rerun()

def data_description_page():
    side_bar()
            
    st.header("Data Description Page")
    if st.button("Describe a column"):
        st.session_state.page = "column_form"
    if st.button("Dataset's Properties"):
        st.table(st.session_state.df.describe(include='all'))
        buf = StringIO()
        st.session_state.df.info(buf=buf)
        df_info = buf.getvalue()

        info_data = []
        sno = 1
        parse_started = False
        for line in df_info.split('\n'):
            if line.startswith('dtypes'):
                parse_started = False
            if parse_started:
                if line.strip():
                    columns = line.split()
                    col_name = columns[1]
                    non_null_count = columns[2]
                    dtype = columns[4]
                    info_data.append([sno, col_name, non_null_count, dtype])
                    sno += 1
            elif line.startswith('---'):
                parse_started = True

        # Creating a DataFrame from the parsed info data
        info_df = pd.DataFrame(info_data, columns=['sno', 'Column', 'Non-Null Count', 'Dtype'])

        # Displaying the DataFrame as a table
        st.table(info_df)
    if st.button("show DataSet"):
        st.table(st.session_state.df)
    # if st.button("Back"):
    #     st.session_state.page = "home"

def column_describe_page():
    side_bar()

    with st.form(key='column_form'):
            st.header('Columns')
            columns = st.session_state.df.columns.to_list()
            col_name = st.session_state.selected_column = st.selectbox("Select a column to normalize", columns)
            submit_button = st.form_submit_button("Submit")
            if submit_button:
                if col_name in st.session_state.df.columns.to_list():
                    st.table(st.session_state.df[col_name].describe())
                else:
                    st.write(f"The column '{col_name}' does not exist in the DataFrame.")


def input():
    st.title("Dataset Preprocessor")
    st.write("Welcome to the Dataset Preprocessor app! This tool helps you preprocess your datasets.")
    
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.session_state.original_df = df.copy()
        st.session_state.df = df
         # Buttons for different actions
        if st.button("continue"):
            st.session_state.page = "home"

def null_val_handle_page():
    side_bar()

    st.header("Handle NULL values")
    if st.button("Show NULL values"):
        st.write(st.session_state.df.isnull().sum()) 
    if st.button("Remove columns"):
        st.session_state.page = "remove_col"
    if st.button("Fill NULL values"):
        st.session_state.page = "fill_null_val"
    if st.button("show DataSet"):
        st.table(st.session_state.df)

def remove_col_page():
    side_bar()

    with st.form(key='remove_col'):
            columns = st.session_state.df.columns.to_list()
            col_name = st.session_state.selected_column = st.selectbox("Select a column to remove it.", columns)
            submit_button = st.form_submit_button("Submit")
            if submit_button:
                if col_name in st.session_state.df.columns.to_list():
                    st.session_state.df = st.session_state.df.drop([col_name], axis=1)
                    st.write(f"The column '{col_name}' Dropped")
                else:
                    st.write(f"The column '{col_name}' does not exist in the DataFrame.")


def fill_null_val():
    side_bar()
    
    with st.form(key='fill_null_values'):
        st.header('Fill NULL values')
        columns = st.session_state.df.columns.to_list()
        col_name = st.session_state.selected_column = st.selectbox("Select a column to FILL", columns)
        fill_method = st.radio("Choose a method to fill NULL values", ("Zero","Mean", "Median", "Mode","Pad","Backfill"))
        submit_button = st.form_submit_button("Submit")
        if submit_button:
            if col_name in st.session_state.df.columns.to_list():
                if fill_method == "Zero":
                    st.session_state.df[col_name].fillna(0, inplace=True)
                elif fill_method == "Mean":
                    st.session_state.df[col_name].fillna(st.session_state.df[col_name].mean(), inplace=True)
                elif fill_method == "Median":
                    st.session_state.df[col_name].fillna(st.session_state.df[col_name].median(), inplace=True)
                elif fill_method == 'Mode':
                    st.session_state.df[col_name].fillna(st.session_state.df[col_name].mode()[0], inplace=True)
                elif fill_method == 'Pad':
                    st.session_state.df[col_name].fillna(method='pad', inplace=True)
                elif fill_method == 'Backfill':
                    st.session_state.df[col_name].fillna(method='bfill', inplace=True)
                st.write(f"The column '{col_name}' Filled")
            else:
                st.write(f"The column '{col_name}' does not exist in the DataFrame.")
def encode_data():
    side_bar()
    st.header("Categorical Columns")
    categorical_columns = st.session_state.df.select_dtypes(include=['object'])
    unique_counts = {col: st.session_state.df[col].nunique() for col in categorical_columns}
    unique_counts_df = pd.DataFrame(list(unique_counts.items()), columns=['Column', 'Unique Count'])
    st.table(unique_counts_df)
    
    categorical_columns = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
    selected_column = st.selectbox("Select a column to encode", categorical_columns)

    if st.button("Submit"):
        st.session_state.df = pd.get_dummies(st.session_state.df, columns=[selected_column],dtype=int)
        st.write(st.session_state.df)
    
    if st.button("show DataSet"):
        st.write(st.session_state.df)


def feature_scale():
    side_bar()

    numeric_columns = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if "normalize_scaler" not in st.session_state:
        st.session_state.normalize_scaler = False
    if "standardize_scaler" not in st.session_state:
        st.session_state.standardize_scaler = False

    button_placeholder_norm = st.empty()
    button_placeholder_stan = st.empty()
    button_placeholder_show = st.empty()
# Create a button inside the placeholder
    button_clicked1 = button_placeholder_norm.button("Normalization (MinMax Scaler)")
    button_clicked2 = button_placeholder_stan.button("Standardization (Standard Scaler)")
    button_clicked3 = button_placeholder_show.button("Show DataSet")
# If the button is clicked, clear the placeholder to hide the button
    if button_clicked1:
        button_placeholder_norm.empty()
        button_placeholder_stan.empty()
        button_placeholder_show.empty()
        if st.button("Normalize whole dataset"):
            scaler = MinMaxScaler()
            st.session_state.df[numeric_columns] = scaler.fit_transform(st.session_state.df[numeric_columns])
            st.table(st.session_state.df)

        st.write("OR")
        st.session_state.selected_column = st.selectbox("Select a column to normalize", numeric_columns)

        if st.button("Normalize a column"):
            scaler = MinMaxScaler()
            st.session_state.df[st.session_state.selected_column] = scaler.fit_transform(st.session_state.df[[st.session_state.selected_column]])
            st.table(st.session_state.df)
        if st.button("back"):
            button_clicked1 = button_placeholder_norm.button("Normalization (MinMax Scaler)")
            button_clicked2 = button_placeholder_stan.button("Standardization (Standard Scaler)")

    if button_clicked2:
        button_placeholder_norm.empty()
        button_placeholder_stan.empty()
        button_placeholder_show.empty()
        if st.button("Standardize whole dataset"):
            scaler = StandardScaler()
            st.session_state.df[numeric_columns] = scaler.fit_transform(st.session_state.df[numeric_columns])
            st.table(st.session_state.df)
        st.write("OR")
        st.session_state.selected_column = st.selectbox("Select a column to standardize", numeric_columns)

        if st.button("Submit"):
            scaler = StandardScaler()
            st.session_state.df[st.session_state.selected_column] = scaler.fit_transform(st.session_state.df[[st.session_state.selected_column]])
            st.table(st.session_state.df)
        if st.button("back"):
            button_clicked1 = button_placeholder_norm.button("Normalization (MinMax Scaler)")
            button_clicked2 = button_placeholder_stan.button("Standardization (Standard Scaler)")
    
    if button_clicked3:
        button_placeholder_norm.empty()
        button_placeholder_stan.empty()
        button_placeholder_show.empty()
        st.table(st.session_state.df)
        if st.button("back"):
            button_clicked1 = button_placeholder_norm.button("Normalization (MinMax Scaler)")
            button_clicked2 = button_placeholder_stan.button("Standardization (Standard Scaler)")

def download():
    side_bar()

    st.header("Download the preprocessed dataset")
    file_name = st.text_input("Enter the file name", "preprocessed_data.csv")

    if st.button("Download CSV"):
        st.write("Downloading CSV...")
        csv = st.session_state.df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to bytes and encode as base64
        href = f'<a href="data:file/csv;base64,{b64}" download="'+file_name+'">Click here Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Dataset Preprocessor",layout="wide")
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
    elif st.session_state.page == "reset":
        reset_page()
    elif st.session_state.page == "confirm_page":
        confirm_page()

if __name__ == "__main__":
    main()
