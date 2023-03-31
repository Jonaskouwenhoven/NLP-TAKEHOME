import bert4srl.bert_utils as util
from torch.nn import CrossEntropyLoss
from transformers import BertForTokenClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
import numpy as np
import argparse


def args_function():
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelpath', '--model_path', help='Path to a fine tuned model', default="saved_models/test")
    parser.add_argument('-devpath', '--test_path', help='Path to Dev Set', default="data\he.conllu")
    parser.add_argument('-epochs', '--epochs', help = "The correct epoch of the fine tuned model", type=int, default=8)
    parser.add_argument('-batchsize', '--batch_size', type=int, default=32)
    parser.add_argument('-learningrate', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-maxlen', '--max_len', type=int, default=256)
    args = parser.parse_args()
    return args


def predictions(data, label):
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index # -100

    model_path = "bert4srl/saved_models/test/EPOCH_8"
    try:
        model, tokenizer = util.load_model(BertForTokenClassification, BertTokenizer, model_path)

    except:
        print("Model not found, please download the model from the google drive link in the README.md file, or ron the MAin_only_Allen.py file")
        exit()
    index2label = util.load_label_dict(f"bert4srl/label2index.json")
    label2index = {v:k for k,v in index2label.items()}
    # data, label = return_sentences()
    # exit()
    prediction_inputs, prediction_mask, prediction_predicate_labels, prediction_labels, prediction_seq_lengths = util.data_to_tensors(data, 
                                                                                                    tokenizer, 
                                                                                                    max_len=256, 
                                                                                                    labels=label, 
                                                                                                    label2index=label2index,
                                                                                                    pad_token_label_id=PAD_TOKEN_LABEL_ID)

    prediction_data = TensorDataset(prediction_inputs, prediction_mask, prediction_labels, prediction_predicate_labels)
    prediction_sampler = RandomSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=32)

    results, preds_list = util.evaluate_bert_model(prediction_dataloader, 32, model, tokenizer, label2index, 
                                                        PAD_TOKEN_LABEL_ID, full_report=True, prefix="Test Set")
    
    return results, preds_list

if __name__ == "__main__":
    print(predictions())