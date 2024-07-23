import boto3
import json
import time
import streamlit as st

def classify_document(text, endpoint_arn, comprehend_client, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = comprehend_client.classify_document(
                Text=text,
                EndpointArn=endpoint_arn
            )
            return response
        except comprehend_client.exceptions.ThrottlingException:
            retries += 1
            time.sleep(2 ** retries)
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    return None

@st.cache_data
def make_predictions(df, endpoint_arn, comprehend_client):
    primary_classes = []
    primary_scores = []
    other_classes = []
    other_scores = []

    st.write("Progress... Please wait...")
    progress_bar = st.progress(0)

    i = 0
    while i < len(df):
        text = df.at(i, 'Nachricht')
        result = classify_document(text, endpoint_arn, comprehend_client)
        if result:
            classes = result.get('Classes', [])
            if classes:
                primary_class = classes[0]['Name']
                primary_score = classes[0]['Score'] * 100
                primary_classes.append(primary_class)
                primary_scores.append(f"{primary_score:.2f}%")
                other_class_names = [cls['Name'] for cls in classes[1:]]
                other_class_scores = [cls['Score'] * 100 for cls in classes[1:]]
                other_classes.append(", ".join(other_class_names))
                other_scores.append(", ".join([f"{score:.2f}%" for score in other_class_scores]))
            else:
                primary_classes.append("")
                primary_scores.append("")
                other_classes.append("")
                other_scores.append("")
            i += 1
        else:
            time.sleep(1)

        progress_bar.progress(i / len(df))

    df['Primary Class'] = primary_classes
    df['Primary Score'] = primary_scores
    df['Other Classes'] = other_classes
    df['Other Scores'] = other_scores
    return df
