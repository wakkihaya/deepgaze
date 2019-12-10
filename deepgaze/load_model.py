import requests, zipfile, io
import itertools
import threading
import time
import sys
import os
import shutil


class LoadModel:
    """Module that allows users to download and load deepgaze pre-trained models. This allows the required 
       models to be downloaded on request when required."""

    def __init__(self):

        self.models = {
            "Head pose estimation": {
                "Link": "https://www.dropbox.com/s/jnra8jt9ty3qp99/head_pose.zip?dl=1",
                "roll": "tensorflow/head_pose/roll/cnn_cccdd_30k.tf",
                "yaw": "tensorflow/head_pose/yaw/cnn_cccdd_30k.tf",
                "pitch": "tensorflow/head_pose/pitch/cnn_cccdd_30k.tf",
            },
            "Haar Cascades": {
                "Link": "https://dl.dropbox.com/s/1a98kz7yrotbpjz/xml.zip",
                "profile face": "xml/haarcascade_profileface.xml",
                "frontal face": "xml/haarcascade_frontalface_alt.xml",
            },
        }

        self.abs_path = os.path.relpath("..") + "/models/"

    def checkExists(self, name):

        """Check if a given model exists.
           args:
               
               name: Name of given model"""

        # Get current directory
        cwd = os.getcwd()
        return os.path.isdir(cwd + "/" + name)

    def getModel(self, name, params=""):
        """Get the file/path of the required model. If not present, download the required
           model.
           
           args:
               
               name: name of given model.
               params: if the given subcategory of model is specified
               
           returns:
               
               return_path: The path of retrieved model
               
        """

        # Check current directory
        cwd = os.getcwd()

        # Check if given model is already downloaded
        if self.checkExists(name) is False:

            zip_file_url = self.models[name]["Link"]

            print("Downloading.... " + name + " model")
            r = requests.get(zip_file_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(name)

        return_path = ""

        # If a given subcategory of model is specified
        if params:

            return_path += cwd + "/" + name + "/" + self.models[name][params]

        else:

            return_path = cwd + "/" + name
            
        print("DONE")

        return return_path
