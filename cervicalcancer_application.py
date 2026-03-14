import base64
import pickle as pk

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu  # keep only if you still want to use it

st.set_page_config(page_title="Cervical Cancer Detection", layout="centered")

train_data_header_names = [
    'Age', 'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD',
    'STDs', 'STDs (number)', 'STDs:condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:genital herpes', 'STDs:HIV',
    'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx'
]


def filedownload(df: pd.DataFrame) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your Predictions</a>'


def parse_float_input(label: str, default_value: str, key: str):
    value_str = st.text_input(label, default_value, key=key).strip()

    if value_str == "":
        st.error(f"{label} cannot be empty.")
        return None

    try:
        return float(value_str)
    except ValueError:
        st.error(f"Invalid input for {label}. Please enter a valid number.")
        return None


@st.cache_resource
def load_single_prediction_model():
    with open("my_saved_CervicalCancer_std_scaler.pkl", "rb") as f:
        return pk.load(f)


@st.cache_resource
def load_multi_prediction_model():
    with open("The_Cervical_Cancer_Model.sav", "rb") as f:
        return pk.load(f)


@st.cache_resource
def load_scaler():
    with open("my_saved_CervicalCancer_std_scaler.pkl", "rb") as f:
        return pk.load(f)


def eligibility_status(givendata):
    loaded_model = load_single_prediction_model()

    input_data_as_numpy_array = np.asarray(givendata, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    new_input_data = pd.DataFrame(input_data_reshaped, columns=train_data_header_names)

    prediction = loaded_model.predict(new_input_data)

    if prediction[0] == 0:
        return "No cervical cancer detected"
    return "Cervical cancer is present"


def main():
    st.header("Cervical Cancer Detection System")

    age = st.slider("Patient age", 0, 120, key="ageslide")
    st.write(f"Patient is {age} years old")

    num_of_pregnancies = st.slider("Number of pregnancies", 0, 30, key="num_pregnancies")

    years_of_hormonal_contraceptives = parse_float_input(
        "How long has the patient been on hormonal contraceptives? (Years)",
        "0",
        "yearsofhormonal_contracept"
    )

    st.write("")

    option2 = st.selectbox("Presence of Intrauterine Device (IUD) in patient", ("", "Yes", "No"), key="IUD")
    iud = 1 if option2 == "Yes" else 0 if option2 == "No" else None

    option3 = st.selectbox("Any STD?", ("", "Yes", "No"), key="patient_std")
    std_status = 1 if option3 == "Yes" else 0 if option3 == "No" else None

    std_number = parse_float_input("Number of STDs", "0", "std_number")

    st.write("")

    option4 = st.selectbox("Is condylomatosis present?", ("", "Yes", "No"), key="the_std_condylomatosis")
    std_condylomatosis = 1 if option4 == "Yes" else 0 if option4 == "No" else None

    option6 = st.selectbox(
        "Any vulvo-perineal condylomatosis?",
        ("", "Yes", "No"),
        key="the_vulvo_perineal_condylomatosis"
    )
    vulvo_perineal_condylomatosis = 1 if option6 == "Yes" else 0 if option6 == "No" else None

    option9 = st.selectbox("Genital herpes?", ("", "Yes", "No"), key="the_genital_herpes")
    genital_herpes = 1 if option9 == "Yes" else 0 if option9 == "No" else None

    option11 = st.selectbox("HIV?", ("", "Yes", "No"), key="the_hiv")
    hiv = 1 if option11 == "Yes" else 0 if option11 == "No" else None

    number_of_diagnosis = parse_float_input("Number of STD diagnoses", "0", "diagnosis_times")

    st.write("")

    option14 = st.selectbox("Diagnosed with cancer?", ("", "Yes", "No"), key="Cancer_diagnosis")
    cancer_diagnosis = 1 if option14 == "Yes" else 0 if option14 == "No" else None

    option15 = st.selectbox(
        "Diagnosed with Cervical Intraepithelial Neoplasia (CIN)?",
        ("", "Yes", "No"),
        key="the_Cin_diagnosis"
    )
    cin_diagnosis = 1 if option15 == "Yes" else 0 if option15 == "No" else None

    option16 = st.selectbox("Diagnosed with HPV?", ("", "Yes", "No"), key="the_diagnosed_HPV")
    diagnosed_hpv = 1 if option16 == "Yes" else 0 if option16 == "No" else None

    option17 = st.selectbox("Other diagnosis?", ("", "Yes", "No"), key="the_Other_diagnosis")
    other_diagnosis = 1 if option17 == "Yes" else 0 if option17 == "No" else None

    if st.button("Predict"):
        missing_fields = []

        if years_of_hormonal_contraceptives is None:
            missing_fields.append("Hormonal contraceptives duration")
        if std_number is None:
            missing_fields.append("Number of STDs")
        if number_of_diagnosis is None:
            missing_fields.append("Number of STD diagnoses")

        select_fields = {
            "IUD": iud,
            "Any STD": std_status,
            "Condylomatosis": std_condylomatosis,
            "Vulvo-perineal condylomatosis": vulvo_perineal_condylomatosis,
            "Genital herpes": genital_herpes,
            "HIV": hiv,
            "Cancer diagnosis": cancer_diagnosis,
            "CIN diagnosis": cin_diagnosis,
            "HPV diagnosis": diagnosed_hpv,
            "Other diagnosis": other_diagnosis,
        }

        for field_name, field_value in select_fields.items():
            if field_value is None:
                missing_fields.append(field_name)

        if missing_fields:
            st.error("Please complete all required fields correctly before prediction.")
            return

        try:
            result = eligibility_status([
                age,
                num_of_pregnancies,
                years_of_hormonal_contraceptives,
                iud,
                std_status,
                std_number,
                std_condylomatosis,
                vulvo_perineal_condylomatosis,
                genital_herpes,
                hiv,
                number_of_diagnosis,
                cancer_diagnosis,
                cin_diagnosis,
                diagnosed_hpv,
                other_diagnosis
            ])
            st.success(result)
        except FileNotFoundError:
            st.error("Model file not found. Please make sure the model file is in the app directory.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def multi(input_data):
    try:
        loaded_model = load_multi_prediction_model()
        std_scaler = load_scaler()

        dfinput = pd.read_csv(input_data)

        st.header("Dataset")
        st.markdown("Preview of the uploaded dataset")
        st.dataframe(dfinput)

        if list(dfinput.columns) != train_data_header_names:
            st.warning("Uploaded CSV column names do not exactly match the expected training columns.")

        dfinput_scaled = std_scaler.transform(dfinput)

        with st.sidebar:
            predict_button = st.button("Click to Predict")

        if predict_button:
            prediction = loaded_model.predict(dfinput_scaled)

            interchange = []
            for i in prediction:
                if i == 0 or i == "0":
                    interchange.append("No Cervical Cancer detected")
                else:
                    interchange.append("Cervical Cancer is present")

            st.subheader("Predicted output")
            prediction_output = pd.Series(interchange, name="Biopsy")
            prediction_id = pd.Series(np.arange(0, len(prediction_output)), name="ID")

            dfresult = pd.concat([prediction_id, prediction_output], axis=1)
            st.dataframe(dfresult)
            st.markdown(filedownload(dfresult), unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
    except pd.errors.ParserError:
        st.error("Unable to read the uploaded CSV file. Please upload a valid CSV.")
    except ValueError as e:
        st.error(f"Data format error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


with st.sidebar:
    selection = st.radio(
        "Choose your prediction system",
        ["Single Prediction", "Multi Prediction"]
    )

if selection == "Multi Prediction":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        multi(uploaded_file)
    else:
        st.info("Waiting for CSV file to be uploaded.")

if selection == "Single Prediction":
    main()
