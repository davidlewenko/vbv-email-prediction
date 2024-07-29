import boto3
import json
import time
import streamlit as st

def classify_document(text, endpoint_arn, comprehend_client, max_retries=10, initial_backoff=1):
    retries = 0
    while retries < max_retries:
        try:
            response = comprehend_client.classify_document(
                Text=text,
                EndpointArn=endpoint_arn
            )
            return response
        except comprehend_client.exceptions.TooManyRequestsException:
            backoff = initial_backoff * (2 ** retries)
            time.sleep(backoff)
            retries += 1
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    return None

@st.cache_data
def make_predictions(df, endpoint_arn, _comprehend_client):
    primary_classes = []
    primary_scores = []
    other_classes = []
    other_scores = []

    st.write("Progress... Please wait...")
    progress_bar = st.progress(0)

    for i, text in enumerate(df['Nachricht']):
        result = classify_document(text, endpoint_arn, _comprehend_client)
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
        else:
            primary_classes.append("")
            primary_scores.append("")
            other_classes.append("")
            other_scores.append("")

        progress_bar.progress((i + 1) / len(df))

    df['Primary Class'] = primary_classes
    df['Primary Score'] = primary_scores
    df['Other Classes'] = other_classes
    df['Other Scores'] = other_scores
    return df
