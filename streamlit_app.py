import os
import boto3
import time
import pandas as pd
import chardet
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO
from streamlit_cognito_auth import CognitoAuthenticator

from utils.comprehend_manager import AWSComprehendManager
from utils.data_processing import preprocess_data, generate_synthetic_data, convert_df
from utils.prediction import make_predictions, make_single_prediction

# Retrieve environment variables
model_arn = os.getenv('MODEL_ARN')
pool_id = os.getenv("POOL_ID")
app_client_id = os.getenv("APP_CLIENT_ID")
app_client_secret = os.getenv("APP_CLIENT_SECRET")
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = "eu-central-1"

# Initialize managers and clients
manager = AWSComprehendManager(region_name=region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
comprehend_client = boto3.client('comprehend', region_name=region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)


# Set Streamlit page configuration
st.set_page_config(page_title="VBV: E-Mail Classifier", layout="centered")

# Initialize session state variable
if 'service_arn' not in st.session_state:
    st.session_state.service_arn = None
if 'service_ready' not in st.session_state:
    st.session_state.service_ready = False

# Initialize Cognito Authenticator
authenticator = CognitoAuthenticator(pool_id=pool_id, app_client_id=app_client_id, app_client_secret=app_client_secret, use_cookies=False)

# Path to the logo imag
logo_path = 'assets/VBV_logo_alternative.png'

# Create a placeholder for the logo
logo_placeholder = st.empty()

# User authenticationn
is_logged_in = authenticator.login()

if not is_logged_in:
    with logo_placeholder.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(logo_path, width=350)
            st.write(' ')
    st.stop()

def logout():
    authenticator.logout()

with st.sidebar:
    st.image(logo_path, use_column_width=True)
    st.text(f"Willkommen,\n{authenticator.get_username()}")
    st.button("Abmelden", "logout_btn", on_click=logout)

def handle_error(error):
    return "Ein unerwarteter Fehler ist aufgetreten. Bitte versuchen Sie es später erneut."

def display_classification_result(result):
    if result:
        classes = result.get('Classes', [])
        if classes:
            primary_class = classes[0]['Name']
            primary_score = classes[0]['Score'] * 100
            st.write(f"Primäre Klasse: {primary_class} ({primary_score:.2f}%)")
            other_class_names = [cls['Name'] for cls in classes[1:]]
            other_class_scores = [cls['Score'] * 100 for cls in classes[1:]]
            st.write(f"Andere Klassen: {', '.join(other_class_names)}")
            st.write(f"Andere Werte: {', '.join([f'{score:.2f}%' for score in other_class_scores])}")
        else:
            st.write("Keine Klassen gefunden.")
    else:
        st.write("Fehler beim Klassifizieren des Textes.")

def show_instructions():
    st.write("""
        Um Ihre CSV-Datei hochzuladen und zu verarbeiten, stellen Sie bitte sicher, dass sie die folgenden Anforderungen erfüllt:
        1. Die Datei muss eine Standard-CSV-Datei sein.
        2. Die Spaltenüberschriften müssen in der ersten Zeile stehen.
        3. Es müssen mindestens zwei Spalten vorhanden sein mit den Namen:
           - **Betreff**
           - **Nachricht**
    """)
    st.write("Sehen Sie sich die synthetische CSV-Datei als Beispiel an oder laden Sie sie herunter.")

def process_uploaded_file(uploaded_file, use_synthetic_data):
    try:
        df = generate_synthetic_data() if use_synthetic_data else read_csv_file(uploaded_file)
        st.write("Hochgeladene CSV-Datei:")
        st.write(df)

        preprocessed_df = preprocess_data(df)
        predicted_df = make_predictions(preprocessed_df, st.session_state.service_arn, comprehend_client)
        st.write("Vorhersagen:")
        st.write(predicted_df)

        display_analysis(predicted_df)

        csv = convert_df(predicted_df)
        st.download_button(label="Daten als CSV herunterladen", data=csv, file_name='vorhersagen.csv', mime='text/csv')
    except Exception as e:
        st.error(handle_error(e))

def read_csv_file(uploaded_file):
    try:
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        uploaded_file.seek(0)
        return pd.read_csv(StringIO(raw_data.decode(encoding)), sep=None, engine='python')
    except Exception as e:
        st.error(handle_error(e))
        return None

def display_analysis(predicted_df):
    analysis = predicted_df['Primary Class'].value_counts()
    st.write("Visualisierung:")
    fig, ax = plt.subplots()
    analysis.plot(kind='bar', ax=ax)
    ax.set_title('Vorhersageanalyse')
    ax.set_xlabel('Kategorie')
    ax.set_ylabel('Anzahl')
    st.pyplot(fig)

def update_service_status(service_arn):
    try:
        status = manager.check_endpoint_status(service_arn)
        status_message = st.empty()
        while status == "CREATING":
            status_message.info("Dienst wird vorbereitet... Dies kann einige Minuten dauern.")
            time.sleep(30)
            status = manager.check_endpoint_status(service_arn)
        if status == "IN_SERVICE":
            status_message.success("Der Dienst ist jetzt aktiv und einsatzbereit.")
            st.session_state.service_ready = True
        else:
            status_message.error("Es gab ein Problem bei der Vorbereitung des Dienstes. Bitte versuchen Sie es später erneut.")
    except Exception as e:
        st.error(handle_error(e))

def check_or_create_service():
    if st.session_state.service_arn is None:
        with st.spinner("Überprüfen oder Erstellen des Klassifizierungsdienstes..."):
            try:
                service_arn = manager.find_active_endpoint()
                if not service_arn:
                    service_arn = manager.create_endpoint(model_arn)
                    if service_arn:
                        st.session_state.service_arn = service_arn
                        st.success(f"Klassifizierungsdienst wird vorbereitet!")
                        update_service_status(service_arn)
                    else:
                        st.error("Fehler beim Vorbereiten des Klassifizierungsdienstes.")
                else:
                    st.session_state.service_arn = service_arn
                    st.session_state.service_ready = True
                    st.success(f"Verwendung des vorhandenen Klassifizierungsdienstes!")
            except Exception as e:
                st.error(handle_error(e))
    else:
        service_arn = st.session_state.service_arn
        update_service_status(service_arn)

check_or_create_service()

if st.session_state.service_ready:
    with st.expander("Freitext eingeben"):
        free_text = st.text_area("Geben Sie hier Ihren Text ein:")
        if st.button("Freitext klassifizieren"):
            if free_text:
                try:
                    result = make_single_prediction(free_text, st.session_state.service_arn, comprehend_client)
                    if result:
                        display_classification_result(result)
                    else:
                        st.write("Fehler beim Klassifizieren des Textes.")
                except Exception as e:
                    st.error(handle_error(e))
            else:
                st.write("Bitte geben Sie einen Text ein.")

    with st.expander("Datei hochladen oder synthetische Daten verwenden"):
        if st.button("Anleitung für CSV-Datei-Upload anzeigen"):
            show_instructions()
        uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei", type="csv")
        use_synthetic_data = st.checkbox("Synthetische Daten verwenden")

        if uploaded_file is not None or use_synthetic_data:
            process_uploaded_file(uploaded_file, use_synthetic_data)