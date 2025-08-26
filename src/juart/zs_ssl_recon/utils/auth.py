import os

import requests
from ray.job_submission import JobSubmissionClient


class AuthJobSubmissionClient(JobSubmissionClient):
    def __init__(
        self,
        address,
        client_id,
        client_secret,
        username,
        password,
        verify=True,
        token_url="",
        introspection_url="",
        verbose=False,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.token_url = token_url
        self.introspection_url = introspection_url
        self.address = address
        self._access_token = None
        self._headers = dict()
        self._verify = verify

        self.generate_token()
        self.validate_token()
        self.check_auth_success()

        super().__init__(address, headers=self._headers, verify=self._verify)

    def submit_job(self, *args, **kwargs):
        self.generate_token()

        return JobSubmissionClient.submit_job(self, *args, **kwargs)

    def list_jobs(self, *args, **kwargs):
        self.generate_token()

        return JobSubmissionClient.list_jobs(self, *args, **kwargs)

    def stop_job(self, *args, **kwargs):
        self.generate_token()

        return JobSubmissionClient.stop_job(self, *args, **kwargs)

    def get_job_status(self, *args, **kwargs):
        self.generate_token()

        return JobSubmissionClient.get_job_status(self, *args, **kwargs)

    def tail_job_logs(self, *args, **kwargs):
        self.generate_token()

        return JobSubmissionClient.tail_job_logs(self, *args, **kwargs)

    def generate_token(self):
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }

        try:
            response = requests.post(self.token_url, data=data)
            response.raise_for_status()  # Raises a HTTPError for bad responses
            self._access_token = response.json().get("access_token")
            self._headers["Authorization"] = f"Bearer {self._access_token}"

            if self._access_token:
                return True
            else:
                print("Token generation failed.")
                return False

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return False

    def validate_token(self):
        try:
            response = requests.post(
                self.introspection_url,
                data={"token": self._access_token, "token_type_hint": "access_token"},
                auth=(self.client_id, self.client_secret),
            )
            response.raise_for_status()
            token_data = response.json()

            if token_data.get("active"):
                return True
            else:
                print("Token is invalid or expired.")
                return False

        except requests.RequestException as e:
            print(f"Validation request failed: {e}")
            return False

    def check_auth_success(self):
        try:
            response = requests.get(
                self.address,
                headers=self._headers,
                allow_redirects=True,
                verify=self._verify,
            )
            is_auth_successful = response.url == os.path.join(self.address, "")

            if is_auth_successful:
                return True
            else:
                print("Authentication failed or redirection did not occur as expected.")
                return False

        except requests.RequestException as e:
            print(f"Authentication check failed: {e}")
            return False
