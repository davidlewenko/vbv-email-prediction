import os
import time
import pandas as pd
import chardet
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO
from streamlit_cognito_auth import CognitoAuthenticator

from utils.comprehend_manager import AWSComprehendManager
from utils.data_processing import preprocess_data, generate_synthetic_data, convert_df
from utils.prediction import make_predictions, classify_document


# Initialize Comprehend Manager
manager = AWSComprehendManager()

# Set Streamlit page configuration
st.set_page_config(page_title="VBV: E-Mail Classifier", layout="centered")

# Retrieve environment variables
model_arn = os.getenv('MODEL_ARN')
pool_id = os.getenv("POOL_ID")
app_client_id = os.getenv("APP_CLIENT_ID")
app_client_secret = os.getenv("APP_CLIENT_SECRET")

# Initialize session state variables
if 'endpoint_arn' not in st.session_state:
    st.session_state.endpoint_arn = None
if 'endpoint_ready' not in st.session_state:
    st.session_state.endpoint_ready = False

# Initialize Cognito Authenticator
authenticator = CognitoAuthenticator(
    pool_id=pool_id,
    app_client_id=app_client_id,
    app_client_secret=app_client_secret,
    use_cookies=False
)

# Path to the logo image
logo_path = 'assets/VBV_logo_alternative.png'

# Create a placeholder for the logo
logo_placeholder = st.empty()

# User authentication
is_logged_in = authenticator.login()

if not is_logged_in:
    # Display the centered logo above the login menu using the placeholder
    with logo_placeholder.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write(' ')
        with col2:
            st.image(logo_path, width=350)
            st.write(' ')  # Adjust spacing if necessary
        with col3:
            st.write(' ')
    st.stop()

def logout():
    authenticator.logout()


# Sidebar with user info and logout button
with st.sidebar:
    st.image(logo_path, use_column_width=True)  # Display logo on logged-in page
    st.text(f"Willkommen,\n{authenticator.get_username()}")
    st.button("Abmelden", "logout_btn", on_click=logout)


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
        predicted_df = make_predictions(preprocessed_df, st.session_state.endpoint_arn)
        st.write("Vorhersagen:")
        st.write(predicted_df)

        display_analysis(predicted_df)

        csv = convert_df(predicted_df)
        st.download_button(label="Daten als CSV herunterladen", data=csv, file_name='vorhersagen.csv', mime='text/csv')
    except pd.errors.ParserError as e:
        st.error(f"Fehler beim Parsen der Datei: {e}")
    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")


def read_csv_file(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    uploaded_file.seek(0)
    return pd.read_csv(StringIO(raw_data.decode(encoding)), sep=None, engine='python')


def display_analysis(predicted_df):
    analysis = predicted_df['Primary Class'].value_counts()
    st.write("Visualisierung:")
    fig, ax = plt.subplots()
    analysis.plot(kind='bar', ax=ax)
    ax.set_title('Vorhersageanalyse')
    ax.set_xlabel('Kategorie')
    ax.set_ylabel('Anzahl')
    st.pyplot(fig)


def update_endpoint_status(endpoint_arn):
    status = manager.check_endpoint_status(endpoint_arn)
    status_message = st.empty()
    while status == "CREATING":
        status_message.info("Endpoint wird erstellt... Das kann einige Minuten in Anspruch nehmen.")
        time.sleep(30)
        status = manager.check_endpoint_status(endpoint_arn)
    if status == "IN_SERVICE":
        status_message.success("Endpoint ist jetzt aktiv und einsatzbereit.")
        st.session_state.endpoint_ready = True
    else:
        status_message.error("Fehler bei der Erstellung des Endpunkts oder unerwarteter Zustand.")


# Function to check and update endpoint status
def check_or_create_endpoint():
    if st.session_state.endpoint_arn is None:
        with st.spinner("Überprüfen oder Erstellen des AWS Comprehend Endpunkts..."):
            endpoint_arn = manager.find_active_endpoint()
            if not endpoint_arn:
                endpoint_arn = manager.create_endpoint(model_arn)
                if endpoint_arn:
                    st.session_state.endpoint_arn = endpoint_arn
                    st.success(f"Endpunkt wird erstellt: {endpoint_arn}")
                    update_endpoint_status(endpoint_arn)
                else:
                    st.error("Fehler beim Erstellen des Endpunkts.")
            else:
                st.session_state.endpoint_arn = endpoint_arn
                st.session_state.endpoint_ready = True
                st.success(f"Verwendung des vorhandenen aktiven Endpunkts: {endpoint_arn}")
    else:
        endpoint_arn = st.session_state.endpoint_arn
        update_endpoint_status(endpoint_arn)


# Check or create endpoint at the start
check_or_create_endpoint()

# Enable free text and file upload options only if the endpoint is active
if st.session_state.endpoint_ready:
    # Free text prediction
    with st.expander("Freitext eingeben"):
        free_text = st.text_area("Geben Sie hier Ihren Text ein:")
        if st.button("Freitext klassifizieren"):
            if free_text:
                result = classify_document(free_text, st.session_state.endpoint_arn)
                display_classification_result(result)
            else:
                st.write("Bitte geben Sie einen Text ein.")

    # File and synthetic data upload
    with st.expander("Datei hochladen oder synthetische Daten verwenden"):
        if st.button("Anleitung für CSV-Datei-Upload anzeigen"):
            show_instructions()
        uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei", type="csv")
        use_synthetic_data = st.checkbox("Synthetische Daten verwenden")

        if uploaded_file is not None or use_synthetic_data:
            process_uploaded_file(uploaded_file, use_synthetic_data)