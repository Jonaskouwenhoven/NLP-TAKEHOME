import pandas as pd
import os
import pandas as pd
from allennlp_models import pretrained
from utils.group9_SRL import predict_total
import re
import string as string_value
allen_models = ["structured-prediction-srl-bert", 'structured-prediction-srl']

MODEL_SRL_BERT = pretrained.load_predictor(allen_models[0])
MODEL_SRL = pretrained.load_predictor(allen_models[1])


def get_constituent(predicate, srl_output):
    
    # get the description of the srl out_put corresponding to the predicate

    for verb in srl_output['verbs']:
        if verb['verb'] == predicate:
            description = verb['description']
            return get_constituent_list(description)
    
            


def get_constituent_list(string):
    # print(string)
    # define a regular expression pattern to match the relevant words
    # pattern = r'\[(ARG\d+|V):\s*([^\]]+)\]|(\b\w+\b)'
    pattern = r"\[([^:]+):\s+([^\]]+)\]"

    # find all matches of the pattern in the string
    matches = re.findall(pattern, string)

    # extract the relevant words from the matches and store them in a list
    words = [match[1] or match[2] for match in matches]

    # append the final period to the list
    if string[-1] in string_value.punctuation:

        words.append(string[-1])

    # print(words)
    return words    


def parse_string(input_string):
    """
    Parses the string output of the SRL model into a dictionary FOUT
    """
    # define a regular expression pattern to match the string format
    # pattern = r"\[(\w+): ([^]]+)\]"
    pattern = r"\[([^:]+):\s+([^\]]+)\]"

        # search for matches using the pattern
    matches = re.findall(pattern, input_string)
    
    # convert the matches into a dictionary
    output_dict = {}
    for match in matches:
        key = match[0]
        value = match[1]
        output_dict[key] = value
    
    return output_dict

def classify(predict, gold):
    """
    Classifies the prediction as correct or incorrect
    input: predict, gold - dictionaries, with keys as labels and values as constituents
    output: total missed, total correct, total incorrect 
    """
    correct = 0
    incorrect = 0
    missed = 0

    for key in gold.keys():
        if key in predict.keys():
            if gold[key].lower() == predict[key].lower():
                correct += 1
            else:
                incorrect += 1
        else:
            missed += 1
    if incorrect > 0:
        return 0
    
    elif missed > 0:
        return 0

    else:
        return 1

def get_description(verb, srl_output):
    """
    Returns the description of the verb
    # """
    # print("Were getting the description")
    # print("The verb is", verb)
    for verb1 in srl_output['verbs']:
        
        if verb1['verb'] == verb:

            # print(verb1['description'])
            return verb1['description']
        
    # print(srl_output)
    return verb1['verb'] # Not correct, but its a temporary fix

def map_list_to_constituent(constituent, output_group_9):
    """
    Maps the output of group 9 to the constituent list
    """
    output_dict = {}

    for i, element in enumerate(output_group_9[1]):
        if element != "_":
            classified_at, label  = output_group_9[0][i], output_group_9[1][i]
            for element1 in constituent:
                if classified_at in element1:
                    output_dict[label] = element1

    return output_dict


def calculate_failure(total, correct):
    """
    Calculates the failure rate
    """
    return (total - correct)/total

def predict_SRL_BERT(text, gold):
    
    predicted_outputs = MODEL_SRL_BERT.predict(sentence = text)

    description = get_description(gold, predicted_outputs)

    return get_constituent_list(description), parse_string(description)


def predict_SRL(text, gold):
        
    predicted_outputs = MODEL_SRL.predict(sentence = text)
    
    
    return parse_string(get_description(gold, predicted_outputs))

def predict_SRL_group9(text, constituent ):

    predicted_outputs = predict_total(text)

    return map_list_to_constituent(constituent, predicted_outputs)
