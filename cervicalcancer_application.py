import base64
import pickle as pk
import shap
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu  # keep only if you still want to use it

st.set_page_config(page_title="CervixAI ", layout="centered")

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
    with open("The_Cervical_Cancer_Model.sav", "rb") as f:
        return pk.load(f)


@st.cache_resource
def load_multi_prediction_model():
    with open("The_Cervical_Cancer_Model.sav", "rb") as f:
        return pk.load(f)


@st.cache_resource
def load_scaler():
    with open("my_saved_CervicalCancer_std_scaler.pkl", "rb") as f:
        return pk.load(f)


@st.cache_resource
def load_shap_background():
    """
    Loads background data used by SHAP.
    Save a small standardized training sample as a CSV or pickle.
    Recommended shape: 50 to 100 rows.
    """
    bg = pd.read_csv("shap_background_data.csv")
    return bg


@st.cache_resource
def load_shap_explainer():
    model = load_single_prediction_model()
    background = load_shap_background()
    explainer = shap.KernelExplainer(model.predict_proba, background)
    return explainer


def safe_flatten_shap_values(shap_values):
    """
    Handles different SHAP output formats.
    Returns 1D SHAP values for positive class.
    """
    if isinstance(shap_values, list):
        return np.array(shap_values[1]).flatten()

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        return shap_values[0, :, 1]
    elif shap_values.ndim == 2:
        return shap_values[0]
    else:
        return shap_values.flatten()


def build_plain_english_explanation(original_row_df, scaled_row_df, positive_only=True, top_n=3):
    """
    original_row_df: one-row DataFrame with original values
    scaled_row_df: one-row DataFrame with standardized values
    """
    model = load_single_prediction_model()
    explainer = load_shap_explainer()

    pred_class = model.predict(scaled_row_df)[0]
    pred_prob = model.predict_proba(scaled_row_df)[0][1]

    shap_values = explainer.shap_values(scaled_row_df)
    patient_shap = safe_flatten_shap_values(shap_values)

    explanation_df = pd.DataFrame({
        "Feature": train_data_header_names,
        "Original Value": original_row_df.iloc[0].values,
        "SHAP Value": patient_shap
    })

    explanation_df["Abs SHAP"] = explanation_df["SHAP Value"].abs()
    explanation_df = explanation_df.sort_values("Abs SHAP", ascending=False)

    positive_drivers = explanation_df[explanation_df["SHAP Value"] > 0].head(top_n)
    negative_drivers = explanation_df[explanation_df["SHAP Value"] < 0].head(top_n)

    if pred_class == 0:
        return f"The model predicts no cervical cancer with probability {1 - pred_prob:.1%}."

    if positive_only:
        if len(positive_drivers) == 0:
            return f"The model predicts cervical cancer with probability {pred_prob:.1%}, but no strong positive drivers were identified."

        phrases = []
        for _, row in positive_drivers.iterrows():
            feature = row["Feature"]
            value = row["Original Value"]
            phrases.append(f"{feature} ({value})")

        joined = ", ".join(phrases[:-1]) + f" and {phrases[-1]}" if len(phrases) > 1 else phrases[0]

        return f"The model predicts cervical cancer with probability {pred_prob:.1%}. The main factors increasing this prediction were {joined}."

    else:
        pos_text = ""
        neg_text = ""

        if len(positive_drivers) > 0:
            pos_parts = [f"{row['Feature']} ({row['Original Value']})" for _, row in positive_drivers.iterrows()]
            pos_text = "The main factors increasing the risk were " + (
                ", ".join(pos_parts[:-1]) + f" and {pos_parts[-1]}" if len(pos_parts) > 1 else pos_parts[0]
            ) + ". "

        if len(negative_drivers) > 0:
            neg_parts = [f"{row['Feature']} ({row['Original Value']})" for _, row in negative_drivers.iterrows()]
            neg_text = "Factors slightly reducing the risk were " + (
                ", ".join(neg_parts[:-1]) + f" and {neg_parts[-1]}" if len(neg_parts) > 1 else neg_parts[0]
            ) + "."

        return f"The model predicts cervical cancer with probability {pred_prob:.1%}. {pos_text}{neg_text}".strip()


def eligibility_status(givendata):
    loaded_model = load_single_prediction_model()
    scaler = load_scaler()

    input_data_as_numpy_array = np.asarray(givendata, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    original_input_df = pd.DataFrame(input_data_reshaped, columns=train_data_header_names)
    scaled_input = scaler.transform(original_input_df)
    scaled_input_df = pd.DataFrame(scaled_input, columns=train_data_header_names)

    prediction = loaded_model.predict(scaled_input_df)

    if prediction[0] == 0:
        return "No cervical cancer detected", None

    explanation = build_plain_english_explanation(
        original_row_df=original_input_df,
        scaled_row_df=scaled_input_df,
        positive_only=True,
        top_n=3
    )
    return "Cervical cancer is present", explanation


def main():
    st.header("CervixAI ")

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
            result, explanation = eligibility_status([
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
            
            if explanation is not None:
                st.subheader("Why the model predicted this")
                st.write(explanation)
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

            scaled_df = pd.DataFrame(dfinput_scaled, columns=train_data_header_names)

            labels = []
            explanations = []

            for idx, pred in enumerate(prediction):
                original_row = dfinput.iloc[[idx]].copy()
                scaled_row = scaled_df.iloc[[idx]].copy()

                if pred == 0 or pred == "0":
                    labels.append("No Cervical Cancer detected")
                    explanations.append("Prediction indicates no cervical cancer.")
                else:
                    labels.append("Cervical Cancer is present")
                    try:
                        explanation = build_plain_english_explanation(
                            original_row_df=original_row,
                            scaled_row_df=scaled_row,
                            positive_only=True,
                            top_n=3
                        )
                    except Exception:
                        explanation = "Positive prediction. Main risk drivers could not be generated."
                    explanations.append(explanation)

            st.subheader("Predicted output")

            prediction_output = pd.Series(labels, name="Biopsy")
            prediction_id = pd.Series(np.arange(0, len(prediction_output)), name="ID")
            explanation_output = pd.Series(explanations, name="Explanation")

            dfresult = pd.concat([prediction_id, prediction_output, explanation_output], axis=1)
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
    st.image("cervical_logo.jpeg")
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
