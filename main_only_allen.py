import pandas as pd
import os
import pandas as pd
from allennlp_models import pretrained
from group9_SRL import predict_total
import re
import string as string_value
# from predict import pred.predict_SRL, pred.predict_SRL_BERT, pred.predict_SRL_group9
import predict as pred
def test_one():
    """
    Testing SRL capability on handeling, statement -> multiple realizations
    """
    
    df = pd.read_json("testdata/test1.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")
    total_srl, total_srl_bert  = 0, 0
    for test in tests[:]:
        original = test['proposition']
        realizations = test['realizations']
        real_srl, real_srl_bert = 1, 1
        for real in realizations:
            sentence = real['realization']
            gold_label = real['propbank_golden_labels']
          
            predicate_gold = gold_label['V']

            
            _, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate_gold)
            output_srl = pred.predict_SRL(sentence, predicate_gold)
            

            real_srl *= pred.classify(output_srl, gold_label)
            real_srl_bert *= pred.classify(output_srl_bert, gold_label)

        total_srl += real_srl
        total_srl_bert += real_srl_bert

    print("SRL score: ", total_srl)
    print("SRL BERT score: ", total_srl_bert)

    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert)

def test_two():
        
    """
    Testing SRL capability on handeling, statement -> Question
    """
    # TODO check if predicted is correct
    df = pd.read_json("testdata/test2.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert= 0, 0
    for test in tests:

        statement = test['statement']
        question = test['question']

        golden_labels = test['propbank_golden_labels'][0]

        statement_gold = golden_labels['Statement']
        question_gold = golden_labels['Question']

        statement_pred = statement_gold['V']
        question_pred = question_gold['V']


        _, output_statement_srl_bert = pred.predict_SRL_BERT(statement, statement_pred)
        output_statement_srl = pred.predict_SRL(statement, statement_pred)


        _, output_question_srl_bert = pred.predict_SRL_BERT(question, question_pred)

        output_question_srl = pred.predict_SRL(question, question_pred)

        SRL_score = pred.classify(output_statement_srl, statement_gold) * pred.classify(output_question_srl, question_gold)

        SRL_BERT_score = pred.classify(output_statement_srl_bert, statement_gold) * pred.classify(output_question_srl_bert, question_gold)


        
        total_srl += SRL_score
        total_srl_bert += SRL_BERT_score

    print("SRL score: ", total_srl)
    print("SRL BERT score: ", total_srl_bert)

    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert)


def test_three():
    """
    Testing SRL capabiltie of handeling, active -> passive
    """
    df = pd.read_json("testdata/test3.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    srl, bert= 0, 0
    for test in tests:
        active = test['active']
        passive = test['passive']

        golden_labels = test['propbank_golden_labels'][0]

        active_gold = golden_labels['Active']
        passive_gold = golden_labels['Passive']

        active_predicate = active_gold['V']
        passive_predicate = passive_gold['V']



        constituent, output_active_srl = pred.predict_SRL_BERT(active, active_predicate)
        output_active_srl_bert = pred.predict_SRL(active, active_gold)


        constituent, output_passive_srl = pred.predict_SRL_BERT(passive, passive_predicate)
        output_passive_srl_bert = pred.predict_SRL(passive, passive_predicate)

        srl += (pred.classify(output_active_srl, active_gold) * pred.classify(output_passive_srl, passive_gold))
        bert += (pred.classify(output_active_srl_bert, active_gold) * pred.classify(output_passive_srl_bert, passive_gold))

    print("SRL score: ", srl)
    print("SRL BERT score: ", bert)

    return pred.calculate_failure(len(tests), srl), pred.calculate_failure(len(tests), bert)


def test_four():
    """
    Testing the SRL model on the test data for Cleft sentences
    """
    df = pd.read_json("testdata/test4.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert= 0, 0
    for test in tests:
        sentence = test['sentence']
        golden_labels = test['propbank_golden_labels']

        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)


        total_srl += pred.classify(output_srl, golden_labels)
        total_srl_bert += pred.classify(output_srl_bert, golden_labels)
    
    print("SRL score: ", total_srl)
    print("SRL BERT score: ", total_srl_bert)

    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert)

def test_five():
        
    """
    Testing SRL capability on handeling, Verb Alternations/Levin's classes (DIR)
    """
    df = pd.read_json("testdata/test5.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert= 0, 0
    for test in tests:

        causative = test['causative']
        inchoative = test['inchoative']

        golden_labels = test['propbank_golden_labels']

        causative_gold = golden_labels['causative']
        inchoative_gold = golden_labels['inchoative']

        causative_pred = causative_gold['V']
        inchoative_pred = inchoative_gold['V']


        constituent, output_causative_srl_bert = pred.predict_SRL_BERT(causative, causative_pred)
        output_causative_srl = pred.predict_SRL(causative, causative_pred)


        constituent, output_inchoative_srl_bert = pred.predict_SRL_BERT(inchoative, inchoative_pred)
        output_inchoative_srl = pred.predict_SRL(inchoative, inchoative_pred)


        SRL_score = pred.classify(output_causative_srl, causative_gold) * pred.classify(output_inchoative_srl, inchoative_gold)

        SRL_BERT_score = pred.classify(output_causative_srl_bert, causative_gold) * pred.classify(output_inchoative_srl_bert, inchoative_gold)



        total_srl += SRL_score
        total_srl_bert += SRL_BERT_score

        

    print("SRL score: ", total_srl)
    print("SRL BERT score: ", total_srl_bert)

    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert)

def test_six():
    """
    Testing SRL capabiltie of handeling, active -> passive
    """
    df = pd.read_json("testdata/test6.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    srl, bert= 0, 0
    for test in tests:
        sentence = test['sentence']
        synonym = test['synonym_sentence']
        golden_labels = test['propbank_golden_labels']

        sentence_gold = golden_labels['Sentence']
        synonym_gold = golden_labels['SynonymSentence']

        sentence_predicate = sentence_gold['V']
        synonym_predicate = synonym_gold['V']



        constituent, output_sentence_srl_bert = pred.predict_SRL_BERT(sentence, sentence_predicate)
        output_sentence_srl = pred.predict_SRL(sentence, sentence_predicate)


        constituent, output_synonym_srl_bert = pred.predict_SRL_BERT(synonym, synonym_predicate)
        output_synonym_srl = pred.predict_SRL(synonym, synonym_predicate)

        srl += (pred.classify(output_sentence_srl, sentence_gold) * pred.classify(output_synonym_srl, synonym_gold))
        bert += (pred.classify(output_sentence_srl_bert, sentence_gold) * pred.classify(output_synonym_srl_bert, synonym_gold))

    print("SRL score: ", srl)
    print("SRL BERT score: ", bert)

    return pred.calculate_failure(len(tests), srl), pred.calculate_failure(len(tests), bert)


def test_seven():
    """
    Testing the SRL model on the test data for Polysemy - Lexical Semantics
    """
    df = pd.read_json("testdata/test7.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert= 0, 0
    for test in tests:
        sentence = test['sentence']
        golden_labels = test['propbank_golden_labels']
        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)

        total_srl += pred.classify(output_srl, golden_labels)
        total_srl_bert += pred.classify(output_srl_bert, golden_labels)


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert)

def test_eight():
    """
    Testing the SRL model on the test data for spelling errors
    """
    df = pd.read_json("testdata/test8.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert = 0, 0
    for test in tests:
        sentence = test['error_sentence']
        golden_labels = test['propbank_golden_labels']

        predicate = golden_labels['V']

        try:
            constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)
            score_srl_bert = pred.classify(output_srl_bert, golden_labels)

        except:
            score_srl_bert = 0


        try:
            output_srl = pred.predict_SRL(sentence, predicate)
            score_srl = pred.classify(output_srl, golden_labels)
        except:
            score_srl = 0

        total_srl += score_srl
        total_srl_bert += score_srl_bert


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert)

def test_nine():
    df = pd.read_json("testdata/test9.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert= 0, 0
    for test in tests:

        sentence = test['sentence']
        golden_labels = test['propbank_golden_labels']
        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)

        total_srl += pred.classify(output_srl, golden_labels)
        total_srl_bert += pred.classify(output_srl_bert, golden_labels)


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert)


if __name__ == "__main__":

    print("######################################")
    print("##  WELCOME TO THE ADVANCED NLP      ##")
    print("##         TAKE HOME EXAM           ##")
    print("######################################")
    print()
    print("This exam has been created by Jonas Kouwenhoven.")
    print("In this Python script, we will be evaluating 2 Semantic Role Labeling (SRL) models on different tests.")

    print("Running Test One: One Proposition -> Multiple Realizations")
    srl1, bert1 = test_one()

    print("Running Test Two: Statement vs. Question")
    srl2, bert2= test_two()

    print("Running Test Three: Active vs. Passive")
    srl3, bert3 = test_three()

    print("Running Test Four: Clefts")
    srl4, bert4= test_four()

    print("Running Test Five: Verb Alternations/Levin's classes")
    srl5, bert5 = test_five()

    print("Running Test Six: Synonyms")
    srl6, bert6 = test_six()

    print("Running Test Seven: Polysemy")
    srl7, bert7= test_seven()

    print("Running Test Eight: Spelling Errors")
    srl8, bert8 = test_eight()

    print("Running Test Nine: Long Distance Dependencies")
    srl9, bert9= test_nine()

    columnnames = ["Test Name", "SRL", "SRL BERT"]

    testnames = ["One Proposition -> Multiple Realizations", "Statement vs. Question", "Active vs. Passive", "Clefts", "Verb Alternations/Levin's classes", "Synonyms", "Polysemy", "Spelling Errors", "Long Distance Dependencies"]
    srl = [srl1, srl2, srl3, srl4, srl5, srl6, srl7, srl8, srl9]
    bert = [bert1, bert2, bert3, bert4, bert5, bert6, bert7, bert8, bert9]

    df = pd.DataFrame(list(zip(testnames, srl, bert)), columns=columnnames)
    df.to_csv("results.csv", index=False)



    # print("Done")