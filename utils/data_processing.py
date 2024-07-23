import pandas as pd
import streamlit as st

def preprocess_data(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if 'Betreff' in df.columns and 'Nachricht' in df.columns:
        df['Nachricht'] = 'Betreff: ' + df['Betreff'].fillna('') + '\n\n' + df['Nachricht'].fillna('')
        df.drop(columns=['Betreff'], inplace=True)
    else:
        st.error("The required columns 'Betreff' and 'Nachricht' are missing from the uploaded file.")
    return df

def generate_synthetic_data():
    data = {
        "Interne Nummer": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "Betreff": [
            "Neue Adresse", "Wohnortänderung", "Pensionierung Anfrage", "Informationen zur Pension",
            "Auszahlung der Abfertigung", "Pensionsantrag Ausland", "Zusammenführung Konten",
            "Übertragung von Vorsorgekassen", "Bankdaten senden", "Dokumente schicken"
        ],
        "Nachricht": [
            "Sehr geehrte Damen und Herren, bitte ändern Sie meine Adresse zu: Neue Straße 12, 12345 Neue Stadt.",
            "Hallo, meine neue Adresse lautet: Alte Straße 34, 54321 Alte Stadt. Bitte aktualisieren.",
            "Ich möchte Informationen über meinen Pensionsantritt am 01.01.2025 erhalten.",
            "Sehr geehrte Damen und Herren, ich habe Fragen zu meiner Pension ab dem 01.01.2024.",
            "Ich bin nach Deutschland gezogen und möchte die Auszahlung meiner Abfertigung beantragen.",
            "Bitte teilen Sie mir mit, wie ich meinen Pensionsantrag von Ausland stellen kann.",
            "Ich möchte meine Konten bei verschiedenen Vorsorgekassen zusammenführen.",
            "Können Sie die Übertragung meines Guthabens von Valida auf VBV durchführen?",
            "Im Anhang finden Sie meine Bankdaten. Bitte bestätigen Sie den Erhalt.",
            "Ich sende Ihnen die angeforderten Dokumente. Bitte prüfen und bestätigen."
        ]
    }
    return pd.DataFrame(data)

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')
