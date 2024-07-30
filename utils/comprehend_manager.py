import boto3
import json
import os
import streamlit as st
from datetime import datetime

class AWSComprehendManager:
    def __init__(self, region_name, aws_access_key_id, aws_secret_access_key):
        self.now = datetime.now().strftime("%Y%m%d%H%M%S")
        self.comprehend_client = boto3.client(
            'comprehend', 
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
            )

    def list_endpoints(self):
        try:
            response = self.comprehend_client.list_endpoints()
            return [ep["EndpointArn"] for ep in response["EndpointPropertiesList"]]
        except Exception as e:
            st.error(f"Error listing endpoints: {str(e)}")
            return []

    def find_active_endpoint(self):
        endpoints = self.list_endpoints()
        for endpoint_arn in endpoints:
            status = self.check_endpoint_status(endpoint_arn)
            if status == "IN_SERVICE":
                return endpoint_arn
        return None

    def create_endpoint(self, model_arn, inference_units=10):
        try:
            response = self.comprehend_client.create_endpoint(
                EndpointName=f"vbv-frontend-endpoint-{self.now}",
                ModelArn=model_arn,
                DesiredInferenceUnits=inference_units,
                Tags=[{"Key": "My1stTag", "Value": "Value1"}]
            )
            return response["EndpointArn"]
        except Exception as e:
            st.error(f"Error creating endpoint: {str(e)}")
            return None

    def check_endpoint_status(self, endpoint_arn):
        try:
            response = self.comprehend_client.describe_endpoint(
                EndpointArn=endpoint_arn
            )
            return response["EndpointProperties"]["Status"]
        except Exception as e:
            st.error(f"Error checking endpoint status: {str(e)}")
            return None
