import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import json
from gluonts.dataset.common import ListDataset
import torch
from pts.model.deepar import DeepAREstimator
from pts import Trainer

# data_id = "de-youtube2021macroV3-gma-ORDERS-2018-12-31-2021-02-01-2020-12-07-splitId-653813_de_2020-12-07_youtube_ON-exo-uid-73996789-390c-48da-a4af-cad4f6581565"

data_id_list = [
    "nl-pytorchTsExpV2-gma-ORDERS-2018-12-31-2021-03-15-2020-12-07-splitId-896672_nl_2020-12-07_youtube_ON-exo-uid-2a9b4681-b5bf-44f7-890d-d56fe2ded2da",
    "de-pytorchTsExpV2-gma-ORDERS-2018-12-31-2021-03-15-2020-12-07-splitId-653813_de_2020-12-07_youtube_ON-exo-uid-ce81a55e-ed30-4364-8d88-5b46bd924b96",
    "dk-pytorchTsExpV2-gma-ORDERS-2018-12-31-2021-03-15-2020-12-07-splitId-950619_dk_2020-12-07_youtube_ON-exo-uid-121c3c9e-9325-436f-846e-169221c9398d",
    "ie-pytorchTsExpV2-gma-ORDERS-2018-12-31-2021-03-15-2020-12-07-splitId-863632_ie_2020-12-07_youtube_ON-exo-uid-48b69ae2-0c6b-413a-8777-1824cd6b7c93",
]

freq = '1D'

prediction_length = 3
context_length = int(prediction_length * 3)

hyperparameters = {
    "time_freq": freq,
    "epochs": "999",
    "early_stopping_patience": "40",
    "mini_batch_size": "128",
    "learning_rate": "5E-4",
    "context_length": str(context_length),
    "prediction_length": str(prediction_length),
    "test_quantiles": [0.16, 0.5, 0.84],
}

TEST_QUANTILES_STR = '["0.16", "0.5", "0.84"]'

def get_relevant_json_str(file_id, data_folder = "rnn-data"):
    # Data Ingestion Step
    config_str = '"configuration": {"output_types": ["mean", "quantiles", "samples"], "quantiles": ' + TEST_QUANTILES_STR  + ', "num_samples": 500}}'

    with open(f"{data_folder}/{file_id}-ts-test.json", 'r') as file:
        json_str = "{\"instances\": ["  + file.read()[:-1].replace('\n', ',') + "]," + config_str

    with open(f"{data_folder}/{file_id}-ts-train.json", 'r') as file:
        json_str_train = "{\"instances\": ["  + file.read()[:-1].replace('\n', ',') + "]," + config_str

    with open(f"{data_folder}/{file_id}-geomap.json", 'r') as file:
        geo_map_str = "{\"instances\": ["  + file.read()[:-1].replace('\n', ',') + "]," + config_str
    
    return json_str, json_str_train, geo_map_str

def ingest_jsonlines_data(data_id):
    json_str, json_str_train, geo_map_str = get_relevant_json_str(data_id)

    json_train_dic = json.loads(json_str_train)
    json_test_dic = json.loads(json_str)
    json_train_instances = json_train_dic["instances"]
    json_test_instances = json_test_dic["instances"]

    train_data_from_jsonlines = list(filter(lambda x: {"start": x["start"], "target": x["target"]}, json_train_instances))
    test_data_from_jsonlines = list(filter(lambda x: {"start": x["start"], "target": x["target"]}, json_test_instances))

    training_data = ListDataset(train_data_from_jsonlines, freq = "1D")
    test_data = ListDataset(test_data_from_jsonlines, freq = "1D")

    return training_data, test_data, geo_map_str

def get_pytorch_ts_json_data(data_id, predictor):
    
    training_data, test_data, geo_map_str = ingest_jsonlines_data(data_id)
    json_str, json_str_train, geo_map_str = get_relevant_json_str(data_id)
    
    json_train_dic = json.loads(json_str_train)
    json_test_dic = json.loads(json_str)

    json_test_dic_ordered = reorder_from_json(json_test_dic)
    json_train_dic_ordered = reorder_from_json(json_train_dic)

    return json_train_dic_ordered, json_test_dic_ordered

def train_deep_ar_model(data_id):
    training_data, test_data, geo_map_str = ingest_jsonlines_data(data_id)
    predictor = train_pytorchts(training_data, test_data, data_id)
    # inference_obj = pkl.load(open(f"inference_output/inference_obj_{data_id}.pkl", "rb"))
    
    # pkl.dump(predictor, open(f"inference_output/predictor_obj_{data_id}.pkl", 'wb'))
    # geo_dict, geo_dict_rev, index_map = get_geo_maps_from_data_id(data_id)
    return predictor

def train_pytorchts(training_data, test_data, data_id = "", nw = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = DeepAREstimator(freq="1D",
                                prediction_length=2,
                                input_size=14,
                                trainer=Trainer(epochs=100, device=device),
                                num_parallel_samples=99)
    
    predictor = estimator.train(training_data = training_data, 
                                validation_data = test_data,
                                num_workers=nw)
    return predictor

if __name__ == "__main__":
    for data_id in data_id_list:
        predictor = train_deep_ar_model(data_id)
        pkl.dump(predictor, open(f"inference_output/predictor_obj_{data_id}.pkl", 'wb'))
    # inference_obj = pkl.load(open(f"inference_output/inference_obj_{data_id}.pkl", "rb"))
    # geo_dict, geo_dict_rev, index_map = get_geo_maps_from_data_id(data_id)
   
