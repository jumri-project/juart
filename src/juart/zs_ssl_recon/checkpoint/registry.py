import io
import os
import time

import pycurl
import torch


class CheckpointRegistry:
    def __init__(
        self, path, project_id, api_v4_url="https://jugit.fz-juelich.de/api/v4"
    ):
        self.url = f"{api_v4_url}/projects/{project_id}/packages/generic/{path}"
        self.token = os.getenv("PRIVATE_TOKEN")

    def save_checkpoint_to_buffer(self, checkpoint):
        tic = time.time()

        buffer = dict()

        for key in checkpoint.keys():
            buffer[key] = io.BytesIO()
            torch.save(checkpoint[key], buffer[key])

        toc = time.time() - tic
        print(f"{current_time()} Saved checkpoint to buffer {toc:.1f} seconds")

        return buffer

    def save_buffer_to_registry(self, buffer):
        tic = time.time()

        for key, data in buffer.items():
            data.seek(0)

            c = pycurl.Curl()
            c.setopt(pycurl.URL, os.path.join(self.url, f"{key}.pth"))
            c.setopt(pycurl.HTTPHEADER, [f"PRIVATE-TOKEN: {self.token}"])
            c.setopt(pycurl.UPLOAD, 1)
            c.setopt(pycurl.READDATA, data)
            c.setopt(
                pycurl.INFILESIZE, len(data.getvalue())
            )  # Get the length of BytesIO data
            c.perform()
            c.close()

        toc = time.time() - tic
        print(
            f"{current_time()} Saved checkpoint to registry {toc:.1f} seconds",
            flush=True,
        )

    def save(self, checkpoint):
        buffer = self.save_checkpoint_to_buffer(checkpoint)
        self.save_buffer_to_registry(buffer)

    def load(self, keys):
        checkpoint = dict()

        for key in keys:
            # Initialize BytesIO object to store file contents
            data = io.BytesIO()

            # Perform the download using pycurl
            c = pycurl.Curl()
            c.setopt(pycurl.URL, os.path.join(self.url, f"{key}.pth"))
            c.setopt(pycurl.HTTPHEADER, [f"PRIVATE-TOKEN: {self.token}"])
            c.setopt(pycurl.WRITEDATA, data)
            c.perform()
            c.close()

            # Reset file_data cursor position to the beginning
            data.seek(0)

            # Store file contents in the buffer dictionary
            checkpoint[key] = torch.load(data, weights_only=True)

        return checkpoint
