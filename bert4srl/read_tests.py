import os
import pandas as pd

def return_sentences():
    path = "Tests/"
    test_file_names = os.listdir(path)
    test1 = pd.read_json(path+test_file_names[0])

    tests = test1['tests']
    correct_sentence, correct_labels = [], []
    for test in tests:
        inputs = test['inputs']
        expected_output = test['expected_output']
        for instance in zip(inputs, expected_output):
            sentence = instance[0]['sentence']
            outputs = instance[1]['semantic_roles']

            # split sentence such that every word is an element also add punctuation

            splited_sentence = sentence.split()
            # last word is word with punctuation
            last_word = splited_sentence[-1]
            # remove punctuation from last word
            last_word = last_word[:-1]
            # add last word without punctuation to list
            splited_sentence[-1] = last_word
            # add punctuation to list
            splited_sentence.append(sentence[-1])
            
            correct_sentence.append(splited_sentence)
            labels = ["_" for _ in range(len(splited_sentence))]
            labels[-1] = "V"
            correct_labels.append(labels)


    return correct_sentence, correct_labels