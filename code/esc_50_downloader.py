import os
import subprocess
from constants import ESC_50_URL, ESC_50_MASTER_ZIP


class ESC50Downloader:
    def __init__(self, download_directory, url=ESC_50_URL):
        self.download_directory = download_directory
        self.url = url
        self.zip_name = ESC_50_MASTER_ZIP
        self.file_path = os.path.join(self.download_directory, self.zip_name)

    def download_and_extract(self):
        if not os.path.isdir(self.download_directory):
            os.makedirs(self.download_directory, exist_ok=True)
            self.download()
            self.extract()

    def download(self):
        try:
            # check=True raises an exception on failure
            subprocess.run(
                ["wget", "-P", self.download_directory, self.url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
        except FileNotFoundError:
            print("Error: wget is not installed. Please install it (e.g., 'sudo apt-get install wget' on Debian/Ubuntu).")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return

    def extract(self):
        try:
            # check=True raises an exception on failure
            subprocess.run(
                ["unzip", self.file_path, '-d', self.download_directory], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error unzipping file: {e}")
        except FileNotFoundError:
            print("Error: The file {} was not found".format(self.file_path))
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
