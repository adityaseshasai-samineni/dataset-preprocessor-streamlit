import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import io
import base64
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def side_bar():
    st.session_state.confirm_reset = False
    with st.sidebar:
        st.title("Dataset Preprocessor")
        if st.button("Data Description"):
            st.session_state.page = "data_description"
        if st.button("Handle Null values"):
            st.session_state.page = "Hand_null_val"
        if st.button("Feature Selection"):
            st.session_state.page = "feature_selection"
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
        if st.button("Documentation"):
            st.session_state.page = "documentation"
def normalize_scaler_page():
    numeric_columns = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if st.button("Normalize whole dataset"):
        scaler = MinMaxScaler()
        st.session_state.df[numeric_columns] = scaler.fit_transform(st.session_state.df[numeric_columns])
        st.write(st.session_state.df)
        logging.info("Whole dataset normalized")
    st.write("OR")
    st.session_state.selected_column = st.selectbox("Select a column to normalize", numeric_columns)
    if st.button("Normalize a column"):
        scaler = MinMaxScaler()
        st.session_state.df[st.session_state.selected_column] = scaler.fit_transform(st.session_state.df[[st.session_state.selected_column]])
        st.write(st.session_state.df)
        logging.info(f"Column \"{st.session_state.selected_column}\" normalized")
    if st.button("back"):
        st.session_state.page = "feature_scale"

def Standardscaler_page():
    numeric_columns = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if st.button("Standardize whole dataset"):
        scaler = StandardScaler()
        st.session_state.df[numeric_columns] = scaler.fit_transform(st.session_state.df[numeric_columns])
        st.write(st.session_state.df)
        logging.info("Whole dataset standardized")
    st.write("OR")
    st.session_state.selected_column = st.selectbox("Select a column to standardize", numeric_columns)
    if st.button("Submit"):
        scaler = StandardScaler()
        st.session_state.df[st.session_state.selected_column] = scaler.fit_transform(st.session_state.df[[st.session_state.selected_column]])
        st.write(st.session_state.df)
        logging.info(f"Column \"{st.session_state.selected_column}\" standardized")
    if st.button("back"):
        st.session_state.page = "feature_scale"

def confirm_page():
    side_bar()
    st.title("Work with another dataset")
    st.write("All the work done on the current dataset will be lost.")
    st.write("Are you sure you want to proceed?")
    if st.button("Yes"):
        st.session_state.page = "input"
        logging.info("Confirmed to work with another dataset")
        st.rerun()
    if st.button("No"):
        st.session_state.page = "home"
        st.rerun()
 
def reset_page():
    side_bar()
    st.title("Reset DataFrame")
    st.write("This action reverses all the operations performed on the DataFrame.")
    st.write("To proceed, enter 'reset' in the text box and click the 'Submit' button.")
    user_input = st.text_input("Enter 'reset' to confirm reset:")
    if st.button("Submit") and user_input == "reset":
        st.session_state.df = st.session_state.original_df.copy()
        logging.info("Performed DataFrame reset")
        st.write("DataFrame has been reset.")

def home_page():
    if st.button("Data Description"):
        st.session_state.page = "data_description"
    if st.button("Handle Null values"):
        st.session_state.page = "Hand_null_val"
    if st.button("Encode Data"):
        st.session_state.page = "encode_data"
    if st.button("Feature Scaling"):
        st.session_state.page = "feature_scale"
    if st.button("Feature Selection"):
        st.session_state.page = "feature_selection"
    if st.button("Work with another dataset"):
        st.session_state.page = "confirm_page"
    if st.button("Documentation"):
        st.session_state.page = "documentation"

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
        logging.info("Dataset's Properties displayed")
        # Creating a DataFrame from the parsed info data
        info_df = pd.DataFrame(info_data, columns=['sno', 'Column', 'Non-Null Count', 'Dtype'])

        # Displaying the DataFrame as a table
        st.table(info_df)
    if st.button("show DataSet"):
        st.write(st.session_state.df)

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
                logging.info(f"Described column \"{col_name}\"")
            else:
                st.write(f"The column '{col_name}' does not exist in the DataFrame.")

def input():
    st.title("Dataset Preprocessor")
    st.write("Welcome to the Dataset Preprocessor app! This tool helps you preprocess your datasets.")
    
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Determine the file type
        file_type = uploaded_file.type

        # Read the file into a DataFrame
        if file_type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        logging.info("\"%s\" uploaded successfully", uploaded_file)
        st.session_state.original_df = df.copy()
        st.session_state.df = df
        # Buttons for different actions
        if st.button("continue"):
            st.session_state.page = "home"

def feature_selection_page():
    side_bar()

    st.header("Feature Selection")
    
    numerical_columns = st.session_state.df.select_dtypes(include=['int','float']).columns.tolist()
    if len(numerical_columns)==0:
        st.write("No numerical columns found in the dataset.")
    else:    
        st.write("Correlation Matrix of the dataset:")
        corr = st.session_state.df[numerical_columns].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr,annot=True,mask=mask, cmap=plt.cm.Reds, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        st.pyplot(f)

        with st.form(key='drop_columns'):
            columns = st.session_state.df.select_dtypes(include=['int','float']).columns.tolist()
            selected_columns = st.multiselect("Select columns to drop", columns)
            submit_button = st.form_submit_button("Submit")
            if submit_button:
                st.session_state.df = st.session_state.df.drop(selected_columns, axis=1)
                logging.info(f"Dropped columns: {', '.join(selected_columns)}")
                st.rerun()

def null_val_handle_page():
    side_bar()

    st.header("Handle NULL values")
    if st.button("Show NULL values"):
        st.write(st.session_state.df.isnull().sum()) 
    if st.button("Remove columns"):
        st.session_state.page = "remove_col"
    if st.button("Drop NULL values"):
        st.session_state.df.dropna(inplace=True)
        st.write("Rows with null values have been dropped.")
        logging.info("NULL values dropped")
    if st.button("Fill NULL values"):
        st.session_state.page = "fill_null_val"
    if st.button("show DataSet"):
        st.write(st.session_state.df)

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
                logging.info(f"Column \"{col_name}\" removed")
            else:
                st.write(f"The column '{col_name}' does not exist in the DataFrame.")
                logging.error(f"Column \"{col_name}\" does not exist in the DataFrame")
            


def fill_null_val():
    side_bar()

    with st.form(key='fill_null_values'):
        st.header('Fill NULL values')
        columns = st.session_state.df.columns.to_list()
        col_name = st.session_state.selected_column = st.selectbox("Select a column to FILL", columns)
        fill_method = st.radio("Choose a method to fill NULL values", ("Zero","Mean", "Median", "Mode","Forwardfill","Backfill"))
        submit_button = st.form_submit_button("Submit")
        numerical_columns = st.session_state.df.select_dtypes(include=['int','float']).columns.tolist()
        if submit_button:
            if col_name in st.session_state.df.columns.to_list():
                if fill_method == 'Forwardfill':
                    st.session_state.df[col_name] = st.session_state.df[col_name].ffill()
                    st.write(f"Filled column \"{col_name}\" with forwardfill")
                    logging.info(f"Filled column \"{col_name}\" with forwardfill")
                elif fill_method == 'Backfill':
                    st.session_state.df[col_name] = st.session_state.df[col_name].bfill()
                    st.write(f"Filled column \"{col_name}\" with backfill")
                    logging.info(f"Filled column \"{col_name}\" with backfill")
                elif fill_method == "Zero" and col_name in numerical_columns:
                    st.session_state.df[col_name].fillna(0, inplace=True)
                    st.write(f"Filled column \"{col_name}\" with 0")
                    logging.info(f"Filled column \"{col_name}\" with 0")
                elif fill_method == "Mean" and col_name in numerical_columns:
                    st.session_state.df[col_name].fillna(st.session_state.df[col_name].mean(), inplace=True)
                    st.write(f"Filled column \"{col_name}\" with mean")
                    logging.info(f"Filled column \"{col_name}\" with mean")
                elif fill_method == "Median" and col_name in numerical_columns:
                    st.session_state.df[col_name].fillna(st.session_state.df[col_name].median(), inplace=True)
                    st.write(f"Filled column \"{col_name}\" with median")
                    logging.info(f"Filled column \"{col_name}\" with median")
                elif fill_method == 'Mode' and col_name in numerical_columns:
                    st.session_state.df[col_name].fillna(st.session_state.df[col_name].mode()[0], inplace=True)
                    st.write(f"Filled column \"{col_name}\" with mode")
                    logging.info(f"Filled column \"{col_name}\" with mode")
                else:
                    st.write(f"The column '{col_name}' is not a numerical column.")            
            else:
                st.write(f"The column '{col_name}' does not exist in the DataFrame.")
                logging.error(f"Column \"{col_name}\" does not exist in the DataFrame")

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
        logging.info(f"Column \"{selected_column}\" encoded")
    if st.button("show DataSet"):
        st.write(st.session_state.df)

def feature_scale():
    side_bar()
    st.header("Feature Scaling")
    if st.button("Normalization (MinMax Scaler)"):
        st.session_state.page = "normalize_scaler"
    if st.button("Standardization (Standard Scaler)"):
        st.session_state.page = "standardize_scaler"
    if st.button("Show DataSet"):
        st.write(st.session_state.df)

def download():
    side_bar()

    st.header("Download the preprocessed dataset")
    file_name = st.text_input("Enter the file name", "preprocessed_data")
    file_format = st.radio("Select file format", ["csv", "xlsx"])

    if st.button("Download"):
        st.write(f"Downloading {file_name}.{file_format}...")
        
        if file_format == "csv":
            csv = st.session_state.df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to bytes and encode as base64
            href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Click here to download CSV file</a>'
        elif file_format == "xlsx":
            towrite = io.BytesIO()
            downloaded_file = st.session_state.df.to_excel(towrite, index=False, engine='openpyxl') 
            towrite.seek(0)  
            b64 = base64.b64encode(towrite.read()).decode() 
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{file_name}.xlsx">Click here to download Excel file</a>'
        
        st.markdown(href, unsafe_allow_html=True)
        logging.info(f"Downloaded the preprocessed dataset as \"{file_name}.{file_format}\"")
        with open('logfile.log', 'r') as f:
            log_text = f.read()

        b64 = base64.b64encode(log_text.encode()).decode()  # Convert log text to bytes and encode as base64
        href = f'<a href="data:file/log;base64,{b64}" download="{file_name}.log">Click here to download the LOG file</a>'
        st.markdown(href, unsafe_allow_html=True)

def documentation_page():
    side_bar()

    st.header("Documentation")

    st.subheader("Upload Your Dataset")
    st.write("You can upload your dataset in CSV or Excel format. The dataset will be read into a DataFrame.")

    st.subheader("Data Description")
    st.write("You can view a description of the dataset or a specific column. The description includes count, mean, std, min, 25%, 50%, 75%, and max for numerical columns and count, unique, top, and freq for categorical columns.")

    st.subheader("Handle Null Values")
    st.write("You can handle null values in the dataset. You can view the null values, remove columns with null values, drop rows with null values, or fill null values with a specific value.")

    st.subheader("Encode Data")
    st.write("You can encode categorical columns in the dataset. The selected column will be replaced with one-hot encoded columns.")

    st.subheader("Feature Scaling")
    st.write("You can scale numerical columns in the dataset. You can normalize the whole dataset or a specific column, or standardize the whole dataset or a specific column.")

    st.subheader("Feature Selection")
    st.write("You can select features in the dataset. You can view a correlation matrix of the dataset and drop selected columns.")

    st.subheader("Download the Dataset")
    st.write("You can download the preprocessed dataset in CSV or Excel format. You can also download the log file containing the actions performed on the dataset.")

    st.subheader("Reset DataFrame")
    st.write("You can reset the DataFrame. This action reverses all the operations performed on the DataFrame.")

    st.subheader("Work with Another Dataset")
    st.write("You can choose to work with another dataset. All the work done on the current dataset will be lost.")

def main():

    logging.basicConfig(filename='logfile.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

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
    elif st.session_state.page == "normalize_scaler":
        normalize_scaler_page()
    elif st.session_state.page == "standardize_scaler":
        Standardscaler_page()
    elif st.session_state.page == "feature_selection":
        feature_selection_page()
    elif st.session_state.page == "documentation":
        documentation_page()

if __name__ == "__main__":
    main()
