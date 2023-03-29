import os
import pandas as pd
from allennlp_models import pretrained
from group9_SRL import predict_total

allen_models = ["structured-prediction-srl-bert", 'structured-prediction-srl']

MODEL_SRL_BERT = pretrained.load_predictor(allen_models[0])
MODEL_SRL = pretrained.load_predictor(allen_models[1])


def predict_SRL(text):
    
    predicted_outputs = MODEL_SRL.predict(sentence = text)

    return predicted_outputs



def predict_SRL_BERT(text):
        
    predicted_outputs = MODEL_SRL_BERT.predict(sentence = text)


    
    return predicted_outputs


def predict_srl_group9(text):

    predicted_outputs = predict_total(text)

    return predicted_outputs
