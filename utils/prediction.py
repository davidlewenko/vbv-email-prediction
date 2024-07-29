import boto3
import json
import time
import streamlit as st

def classify_documents(text_list, endpoint_arn, _comprehend_client, max_retries=10, initial_backoff=1):
    retries = 0
    while retries < max_retries:
        try:
            responses = []
            for text in text_list:
                response = _comprehend_client.classify_document(
                    Text=text,
                    EndpointArn=endpoint_arn
                )
                responses.append(response)
            return responses
        except _comprehend_client.exceptions.TooManyRequestsException:
            backoff = initial_backoff * (2 ** retries)
            time.sleep(backoff)
            retries += 1
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
    return None

def retry_missing_predictions(df, endpoint_arn, _comprehend_client, max_retries=5):
    missing_indices = df[df['Primary Class'].isna()].index.tolist()
    retries = 0
    
    while missing_indices and retries < max_retries:
        missing_texts = df.loc[missing_indices, 'Nachricht'].tolist()
        results = classify_documents(missing_texts, endpoint_arn, _comprehend_client)
        
        if results:
            for i, idx in enumerate(missing_indices):
                result = results[i]
                if result:
                    classes = result.get('Classes', [])
                    if classes:
                        primary_class = classes[0]['Name']
                        primary_score = classes[0]['Score'] * 100
                        df.at[idx, 'Primary Class'] = primary_class
                        df.at[idx, 'Primary Score'] = f"{primary_score:.2f}%"
                        other_class_names = [cls['Name'] for cls in classes[1:]]
                        other_class_scores = [cls['Score'] * 100 for cls in classes[1:]]
                        df.at[idx, 'Other Classes'] = ", ".join(other_class_names)
                        df.at[idx, 'Other Scores'] = ", ".join([f"{score:.2f}%" for score in other_class_scores])
        
        missing_indices = df[df['Primary Class'].isna()].index.tolist()
        retries += 1
    
    return df

@st.cache_data
def make_predictions(df, endpoint_arn, _comprehend_client, batch_size=25):
    primary_classes = [""] * len(df)
    primary_scores = [""] * len(df)
    other_classes = [""] * len(df)
    other_scores = [""] * len(df)

    st.write("Progress... Please wait...")
    progress_bar = st.progress(0)
    total_batches = (len(df) // batch_size) + 1

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        batch_texts = df['Nachricht'].iloc[start_idx:end_idx].tolist()

        if not batch_texts:
            continue

        results = classify_documents(batch_texts, endpoint_arn, _comprehend_client)

        if results:
            for i, result in enumerate(results):
                if result:
                    classes = result.get('Classes', [])
                    if classes:
                        primary_class = classes[0]['Name']
                        primary_score = classes[0]['Score'] * 100
                        primary_classes[start_idx + i] = primary_class
                        primary_scores[start_idx + i] = f"{primary_score:.2f}%"
                        other_class_names = [cls['Name'] for cls in classes[1:]]
                        other_class_scores = [cls['Score'] * 100 for cls in classes[1:]]
                        other_classes[start_idx + i] = ", ".join(other_class_names)
                        other_scores[start_idx + i] = ", ".join([f"{score:.2f}%" for score in other_class_scores])
                else:
                    primary_classes[start_idx + i] = ""
                    primary_scores[start_idx + i] = ""
                    other_classes[start_idx + i] = ""
                    other_scores[start_idx + i] = ""

        progress_bar.progress((batch_num + 1) / total_batches)

    df['Primary Class'] = primary_classes
    df['Primary Score'] = primary_scores
    df['Other Classes'] = other_classes
    df['Other Scores'] = other_scores

    # Retry missing predictions
    df = retry_missing_predictions(df, endpoint_arn, _comprehend_client)

    return df
