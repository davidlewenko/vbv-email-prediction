import boto3
import json
import streamlit as st
from datetime import datetime

class AWSComprehendManager:
    def __init__(self):
        self.now = datetime.now().strftime("%Y%m%d%H%M%S")
        self.comprehend_client = boto3.client('comprehend')

    def list_endpoints(self):
        """
        List existing AWS Comprehend endpoints using boto3.

        Returns:
        list: A list of endpoint ARNs.
        """
        try:
            response = self.comprehend_client.list_endpoints()
            return [ep["EndpointArn"] for ep in response["EndpointPropertiesList"]]
        except Exception as e:
            st.error(f"Error listing endpoints: {str(e)}")
            return []

    def find_active_endpoint(self):
        """
        Find an active AWS Comprehend endpoint.

        Returns:
        str: The ARN of an active endpoint, or None if no active endpoint is found.
        """
        endpoints = self.list_endpoints()
        for endpoint_arn in endpoints:
            status = self.check_endpoint_status(endpoint_arn)
            if status == "IN_SERVICE":
                return endpoint_arn
        return None

    def create_endpoint(self, model_arn, inference_units=1):
        """
        Create an AWS Comprehend endpoint using boto3.

        Args:
        model_arn (str): The ARN of the Comprehend model.
        inference_units (int): The number of inference units for the endpoint.

        Returns:
        str: The ARN of the created endpoint.
        """
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
        """
        Check the status of an AWS Comprehend endpoint using boto3.

        Args:
        endpoint_arn (str): The ARN of the Comprehend endpoint.

        Returns:
        str: The status of the endpoint.
        """
        try:
            response = self.comprehend_client.describe_endpoint(
                EndpointArn=endpoint_arn
            )
            return response["EndpointProperties"]["Status"]
        except Exception as e:
            st.error(f"Error checking endpoint status: {str(e)}")
            return None
