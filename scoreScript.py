# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"Atr1": pd.Series([0], dtype="int64"), "Atr2": pd.Series([0], dtype="int64"), "Atr3": pd.Series([0], dtype="int64"), "Atr4": pd.Series([0], dtype="int64"), "Atr5": pd.Series([0], dtype="int64"), "Atr6": pd.Series([0], dtype="int64"), "Atr7": pd.Series([0], dtype="int64"), "Atr8": pd.Series([0], dtype="int64"), "Atr9": pd.Series([0], dtype="int64"), "Atr10": pd.Series([0], dtype="int64"), "Atr11": pd.Series([0], dtype="int64"), "Atr12": pd.Series([0], dtype="int64"), "Atr13": pd.Series([0], dtype="int64"), "Atr14": pd.Series([0], dtype="int64"), "Atr15": pd.Series([0], dtype="int64"), "Atr16": pd.Series([0], dtype="int64"), "Atr17": pd.Series([0], dtype="int64"), "Atr18": pd.Series([0], dtype="int64"), "Atr19": pd.Series([0], dtype="int64"), "Atr20": pd.Series([0], dtype="int64"), "Atr21": pd.Series([0], dtype="int64"), "Atr22": pd.Series([0], dtype="int64"), "Atr23": pd.Series([0], dtype="int64"), "Atr24": pd.Series([0], dtype="int64"), "Atr25": pd.Series([0], dtype="int64"), "Atr26": pd.Series([0], dtype="int64"), "Atr27": pd.Series([0], dtype="int64"), "Atr28": pd.Series([0], dtype="int64"), "Atr29": pd.Series([0], dtype="int64"), "Atr30": pd.Series([0], dtype="int64"), "Atr31": pd.Series([0], dtype="int64"), "Atr32": pd.Series([0], dtype="int64"), "Atr33": pd.Series([0], dtype="int64"), "Atr34": pd.Series([0], dtype="int64"), "Atr35": pd.Series([0], dtype="int64"), "Atr36": pd.Series([0], dtype="int64"), "Atr37": pd.Series([0], dtype="int64"), "Atr38": pd.Series([0], dtype="int64"), "Atr39": pd.Series([0], dtype="int64"), "Atr40": pd.Series([0], dtype="int64"), "Atr41": pd.Series([0], dtype="int64"), "Atr42": pd.Series([0], dtype="int64"), "Atr43": pd.Series([0], dtype="int64"), "Atr44": pd.Series([0], dtype="int64"), "Atr45": pd.Series([0], dtype="int64"), "Atr46": pd.Series([0], dtype="int64"), "Atr47": pd.Series([0], dtype="int64"), "Atr48": pd.Series([0], dtype="int64"), "Atr49": pd.Series([0], dtype="int64"), "Atr50": pd.Series([0], dtype="int64"), "Atr51": pd.Series([0], dtype="int64"), "Atr52": pd.Series([0], dtype="int64"), "Atr53": pd.Series([0], dtype="int64"), "Atr54": pd.Series([0], dtype="int64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
