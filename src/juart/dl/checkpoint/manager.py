import datetime
import io
import multiprocessing
import os
import time

import torch
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem


def current_time(format="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.now().strftime(format)


def ensure_directory_exists(fs, directory):
    if not fs.exists(directory):
        fs.mkdir(directory, create_parents=True)


class CheckpointManager:
    def __init__(
        self,
        directory,
        root_dir="",
        endpoint_url="https://s3.fz-juelich.de",
        backend="local",
    ):
        self.directory = directory

        if backend == "local":
            self.fs = DirFileSystem(
                root_dir,
                LocalFileSystem(),
            )

        elif backend == "s3":
            # Environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set.
            self.fs = S3FileSystem(
                endpoint_url=endpoint_url,
                default_fill_cache=False,
                default_cache_type=None,
            )

        ctx = multiprocessing.get_context("spawn")

        self.save_queue = ctx.Queue()
        self.save_lock = ctx.Lock()
        self.save_process = ctx.Process(target=self.save_checkpoint_process)
        # Daemonize the process so it automatically exits when the main process exits
        self.save_process.daemon = True
        self.save_process.start()

    def save_checkpoint_to_buffer(self, checkpoint):
        tic = time.time()

        buffer = dict()

        for key in checkpoint.keys():
            buffer[key] = io.BytesIO()
            torch.save(checkpoint[key], buffer[key])

        toc = time.time() - tic
        print(f"{current_time()} Saved checkpoint to buffer {toc:.1f} seconds")

        return buffer

    def save_buffer_to_filesystem(self, buffer, tag):
        tic = time.time()

        ensure_directory_exists(self.fs, f"{self.directory}{tag}")

        with self.fs.transaction:
            for key in buffer.keys():
                with self.fs.open(
                    os.path.join(f"{self.directory}{tag}", f"{key}.pth"), mode="wb"
                ) as f:
                    f.write(buffer[key].getvalue())

        toc = time.time() - tic
        print(
            f"{current_time()} Saved buffer to filesystem in {toc:.1f} seconds",
            flush=True,
        )

    def save_checkpoint_process(self):
        while True:
            self.save_buffer_to_filesystem(*self.save_queue.get())
            self.save_lock.release()
            print(f"{current_time()} Completed saving checkpoint.", flush=True)

    def save(self, checkpoint, tag="", block=True):
        if self.save_lock.acquire(block=block):
            print(f"{current_time()} Schedule checkpoint save with tag: {tag} ...")
            self.save_queue.put((self.save_checkpoint_to_buffer(checkpoint), tag))

    def release(self):
        if self.save_lock.acquire(block=True):
            self.save_lock.release()

    def load(self, keys, tag="", map_location=None):
        # Force a refresh of the directory listing to
        # ensure the filesystem's cache is up-to-date
        self.ls(refresh=True)

        checkpoint = dict()

        for key in keys:
            if self.fs.isfile(os.path.join(f"{self.directory}{tag}", f"{key}.pth")):
                with self.fs.open(
                    os.path.join(f"{self.directory}{tag}", f"{key}.pth"), mode="rb"
                ) as f:
                    checkpoint[key] = torch.load(
                        f, map_location=map_location, weights_only=True
                    )

            else:
                checkpoint[key] = None

        return checkpoint

    def ls(self, refresh=False, detail=False):
        if self.fs.isdir(self.directory):
            return self.fs.ls(self.directory, refresh=refresh, detail=detail)
