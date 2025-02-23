import subprocess
import os
from constants import (FASHION_MNIST_TRAIN_IMAGES_URL, FASHION_MNIST_TRAIN_LABELS_URL,
                       FASHION_MNIST_TEST_IMAGES_URL, FASHION_MNIST_TEST_LABELS_URL)


class FashionMNISTDownloader:
    def __init__(self, download_directory):
        self.download_directory = download_directory

    def download(self):
        if not os.path.isdir(self.download_directory):
            os.makedirs(self.download_directory, exist_ok=True)

            self.download_file(self.download_directory,
                               FASHION_MNIST_TRAIN_IMAGES_URL)
            self.download_file(self.download_directory,
                               FASHION_MNIST_TRAIN_LABELS_URL)
            self.download_file(self.download_directory,
                               FASHION_MNIST_TEST_IMAGES_URL)
            self.download_file(self.download_directory,
                               FASHION_MNIST_TEST_LABELS_URL)

        return

    def download_file(self, download_directory, url):
        try:
            # check=True raises an exception on failure
            subprocess.run(["wget", "-P", download_directory, url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
        except FileNotFoundError:
            print("Error: wget is not installed. Please install it (e.g., 'sudo apt-get install wget' on Debian/Ubuntu).")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return
