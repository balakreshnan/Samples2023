import os
import logging
import json
import numpy
from typing import  Dict
from transformers.pipelines.audio_utils import ffmpeg_read
import whisper
import torch
import shutil
import base64

SAMPLE_RATE = 16000

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # deserialize the model file back into a sklearn model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), 'vca_whisper.pkl')

    model = whisper.load_model("large-v2", download_root=os.getenv("AZUREML_MODEL_DIR"))
    logging.info("Init complete")


def run(data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    #logging.info(data)
    inputs = base64.b64decode(data)
    audio_nparray = ffmpeg_read(inputs, SAMPLE_RATE)
    audio_tensor= torch.from_numpy(audio_nparray)
    
    # run inference pipeline
    result = model.transcribe(audio_nparray)

    # postprocess the prediction
    return {"text": result["text"]}
