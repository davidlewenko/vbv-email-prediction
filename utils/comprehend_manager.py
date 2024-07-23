import subprocess
import json
import streamlit as st
from datetime import datetime

class AWSComprehendManager:
    def __init__(self):
        self.now = datetime.now().strftime("%Y%m%d%H%M%S")

    def list_endpoints(self):
        """
        List existing AWS Comprehend endpoints using AWS CLI.

        Returns:
        list: A list of endpoint ARNs.
        """
        try:
            result = subprocess.run(
                ["aws", "comprehend", "list-endpoints"],
                capture_output=True,
                text=True,
                check=True
            )
            response = json.loads(result.stdout)
            return [ep["EndpointArn"] for ep in response["EndpointPropertiesList"]]
        except subprocess.CalledProcessError as e:
            st.error(f"Error listing endpoints: {e.stderr}")
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
        Create an AWS Comprehend endpoint using AWS CLI.

        Args:
        model_arn (str): The ARN of the Comprehend model.
        inference_units (int): The number of inference units for the endpoint.

        Returns:
        str: The ARN of the created endpoint.
        """
        try:
            result = subprocess.run(
                [
                    "aws", "comprehend", "create-endpoint",
                    "--desired-inference-units", str(inference_units),
                    "--endpoint-name", f"vbv-frontend-endpoint-{self.now}",
                    "--model-arn", model_arn,
                    "--tags", "Key=My1stTag,Value=Value1"
                ],
                capture_output=True,
                text=True,
                check=True
            )
            response = json.loads(result.stdout)
            return response["EndpointArn"]
        except subprocess.CalledProcessError as e:
            st.error(f"Error creating endpoint: {e.stderr}")
            return None

    def check_endpoint_status(self, endpoint_arn):
        """
        Check the status of an AWS Comprehend endpoint using AWS CLI.

        Args:
        endpoint_arn (str): The ARN of the Comprehend endpoint.

        Returns:
        str: The status of the endpoint.
        """
        try:
            result = subprocess.run(
                [
                    "aws", "comprehend", "describe-endpoint",
                    "--endpoint-arn", endpoint_arn
                ],
                capture_output=True,
                text=True,
                check=True
            )
            response = json.loads(result.stdout)
            return response["EndpointProperties"]["Status"]
        except subprocess.CalledProcessError as e:
            st.error(f"Error checking endpoint status: {e.stderr}")
            return None
