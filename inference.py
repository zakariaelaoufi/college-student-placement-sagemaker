import joblib
import json
import os
import numpy as np
from io import StringIO


def model_fn(model_dir):
    """Load the trained model for inference."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if isinstance(data, dict):
            data = [data[key] for key in data.keys()]
        return np.array(data).reshape(1, -1)
    elif request_content_type == 'text/csv':
        return np.loadtxt(StringIO(request_body), delimiter=",")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction.tolist())
    elif response_content_type == 'text/csv':
        return ",".join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")


def predict_fn(input_data, model):
    """Run prediction using the trained model."""
    return model.predict(input_data)