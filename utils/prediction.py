# prediction.py

import time
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

def classify_documents(text_list, endpoint_arn, _comprehend_client, max_retries=10, initial_backoff=2):
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
            st.write(f"[Retry {retries + 1}/{max_retries}] Rate limit exceeded, retrying in {backoff} seconds...")
            time.sleep(backoff)
            retries += 1
        except (BotoCoreError, ClientError) as e:
            st.write(f"Error during document classification: {str(e)}")
            return None
    st.write("Max retries reached, some rows may not be predicted.")
    return None

def retry_missing_predictions(df, endpoint_arn, _comprehend_client, max_retries=10, initial_backoff=2):
    attempt = 0
    while True:
        missing_indices = df[df['Primary Class'].isna()].index.tolist()
        if not missing_indices:
            st.write("All rows have been successfully predicted.")
            break

        attempt += 1
        st.write(f"Attempting to predict {len(missing_indices)} missing entries, attempt {attempt}")
        missing_texts = df.loc[missing_indices, 'Nachricht'].tolist()
        results = classify_documents(missing_texts, endpoint_arn, _comprehend_client, max_retries=max_retries, initial_backoff=initial_backoff)
        
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
        else:
            st.write(f"Error occurred while retrying missing predictions on attempt {attempt}.")

        # Wait before the next attempt to avoid immediate retry
        time.sleep(initial_backoff * (2 ** attempt))

    return df

@st.cache_data
def make_predictions(df, endpoint_arn, _comprehend_client, batch_size=10):  # Reduced batch size
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

        else:
            st.write(f"Batch {batch_num + 1}/{total_batches} failed. Retrying...")
            missing_texts = df['Nachricht'].iloc[start_idx:end_idx].tolist()
            missing_results = classify_documents(missing_texts, endpoint_arn, _comprehend_client)
            if missing_results:
                for i, result in enumerate(missing_results):
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

    # Retry missing predictions until all are processed
    df = retry_missing_predictions(df, endpoint_arn, _comprehend_client)

    return df
