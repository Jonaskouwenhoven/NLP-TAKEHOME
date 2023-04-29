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
    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests[:]:
        original = test['proposition']
        realizations = test['realizations']
        real_srl, real_srl_bert, real_group_9 = 1, 1, 1
        for real in realizations:
            sentence = real['realization']
            gold_label = real['propbank_golden_labels']
          
            predicate_gold = gold_label['V']

            
            constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate_gold)
            output_srl = pred.predict_SRL(sentence, predicate_gold)


            output_group_9 = pred.predict_SRL_group9(sentence, constituent)
            

            real_srl *= pred.classify(output_srl, gold_label)
            real_srl_bert *= pred.classify(output_srl_bert, gold_label)
            real_group_9 *= pred.classify(output_group_9, gold_label)

        total_srl += real_srl
        total_srl_bert += real_srl_bert
        total_group_9 += real_group_9


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)

def test_two():
        
    """
    Testing SRL capability on handeling, statement -> Question
    """
    # TODO check if predicted is correct
    df = pd.read_json("testdata/test2.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:

        statement = test['statement']
        question = test['question']

        golden_labels = test['propbank_golden_labels'][0]

        statement_gold = golden_labels['Statement']
        question_gold = golden_labels['Question']

        statement_pred = statement_gold['V']
        question_pred = question_gold['V']


        constituent, output_statement_srl_bert = pred.predict_SRL_BERT(statement, statement_pred)
        output_statement_srl = pred.predict_SRL(statement, statement_pred)
        output_statement_group_9 = pred.predict_SRL_group9(statement, constituent)


        constituent, output_question_srl_bert = pred.predict_SRL_BERT(question, question_pred)

        output_question_srl = pred.predict_SRL(question, question_pred)
        output_question_group_9 = pred.predict_SRL_group9(question, constituent)

        SRL_score = pred.classify(output_statement_srl, statement_gold) * pred.classify(output_question_srl, question_gold)

        SRL_BERT_score = pred.classify(output_statement_srl_bert, statement_gold) * pred.classify(output_question_srl_bert, question_gold)

        SRL_group_9_score = pred.classify(output_statement_group_9, statement_gold) * pred.classify(output_question_group_9, question_gold)

        
        total_srl += SRL_score
        total_srl_bert += SRL_BERT_score
        total_group_9 += SRL_group_9_score


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)


def test_three():
    """
    Testing SRL capabiltie of handeling, active -> passive
    """
    df = pd.read_json("testdata/test3.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    srl, bert, group9 = 0, 0, 0
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
        output_active_group_9 = pred.predict_SRL_group9(active, constituent)


        constituent, output_passive_srl = pred.predict_SRL_BERT(passive, passive_predicate)
        output_passive_srl_bert = pred.predict_SRL(passive, passive_predicate)
        output_passive_group_9 = pred.predict_SRL_group9(passive, constituent)


        srl += (pred.classify(output_active_srl, active_gold) * pred.classify(output_passive_srl, passive_gold))
        bert += (pred.classify(output_active_srl_bert, active_gold) * pred.classify(output_passive_srl_bert, passive_gold))
        group9 += (pred.classify(output_active_group_9, active_gold) * pred.classify(output_passive_group_9, passive_gold))


    return pred.calculate_failure(len(tests), srl), pred.calculate_failure(len(tests), bert), pred.calculate_failure(len(tests), group9)


def test_four():
    """
    Testing the SRL model on the test data for Cleft sentences
    """
    df = pd.read_json("newtestdata/4_instruments.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:
        sentence = test['original_sentence']
        print(sentence)
        golden_labels = test['propbank_golden_labels']

        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)
        output_group_9 = pred.predict_SRL_group9(sentence, constituent)

        print("srl", output_srl)
        print("bert", output_srl_bert)
        print("group0", output_group_9)
        print(" ")
        total_srl += pred.classify(output_srl, golden_labels)
        total_srl_bert += pred.classify(output_srl_bert, golden_labels)
        total_group_9 += pred.classify(output_group_9, golden_labels)
    

    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)


def test_five():
    """
    Testing the SRL model on the test data for Cleft sentences
    """
    df = pd.read_json("newtestdata/5_context.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:
        sentence = test['original_sentence']
        print(sentence)
        golden_labels = test['propbank_golden_labels']

        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)
        output_group_9 = pred.predict_SRL_group9(sentence, constituent)

        print("srl", output_srl)
        print("bert", output_srl_bert)
        print("group0", output_group_9)
        print(" ")
        total_srl += pred.classify(output_srl, golden_labels)
        total_srl_bert += pred.classify(output_srl_bert, golden_labels)
        total_group_9 += pred.classify(output_group_9, golden_labels)
    

    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)

# def test_five():
        
#     """
#     Testing SRL capability on handeling, Verb Alternations/Levin's classes (DIR)
#     """
#     df = pd.read_json("testdata/test5.json")
#     tests = df['tests']
#     print(f"Number of tests: {len(tests)}")

#     total_srl, total_srl_bert, total_group_9 = 0, 0, 0
#     for test in tests:

#         causative = test['causative']
#         inchoative = test['inchoative']

#         golden_labels = test['propbank_golden_labels']

#         causative_gold = golden_labels['causative']
#         inchoative_gold = golden_labels['inchoative']

#         causative_pred = causative_gold['V']
#         inchoative_pred = inchoative_gold['V']


#         constituent, output_causative_srl_bert = pred.predict_SRL_BERT(causative, causative_pred)
#         output_causative_srl = pred.predict_SRL(causative, causative_pred)
#         output_causative_group_9 = pred.predict_SRL_group9(causative, constituent)


#         constituent, output_inchoative_srl_bert = pred.predict_SRL_BERT(inchoative, inchoative_pred)
#         output_inchoative_srl = pred.predict_SRL(inchoative, inchoative_pred)
#         output_inchoative_group_9 = pred.predict_SRL_group9(inchoative, constituent)


#         SRL_score = pred.classify(output_causative_srl, causative_gold) * pred.classify(output_inchoative_srl, inchoative_gold)

#         SRL_BERT_score = pred.classify(output_causative_srl_bert, causative_gold) * pred.classify(output_inchoative_srl_bert, inchoative_gold)

#         SRL_group_9_score = pred.classify(output_causative_group_9, causative_gold) * pred.classify(output_inchoative_group_9, inchoative_gold)


#         total_srl += SRL_score
#         total_srl_bert += SRL_BERT_score
#         total_group_9 += SRL_group_9_score

        

#     return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)

def test_six():
    """
    Testing SRL capabiltie of handeling, active -> passive
    """
    df = pd.read_json("testdata/test6.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    srl, bert, group9 = 0, 0, 0
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
        output_sentence_group_9 = pred.predict_SRL_group9(sentence, constituent)


        constituent, output_synonym_srl_bert = pred.predict_SRL_BERT(synonym, synonym_predicate)
        output_synonym_srl = pred.predict_SRL(synonym, synonym_predicate)
        output_synonym_group_9 = pred.predict_SRL_group9(synonym, constituent)

        srl += (pred.classify(output_sentence_srl, sentence_gold) * pred.classify(output_synonym_srl, synonym_gold))
        bert += (pred.classify(output_sentence_srl_bert, sentence_gold) * pred.classify(output_synonym_srl_bert, synonym_gold))
        group9 += (pred.classify(output_sentence_group_9, sentence_gold) * pred.classify(output_synonym_group_9, synonym_gold))


    return pred.calculate_failure(len(tests), srl), pred.calculate_failure(len(tests), bert), pred.calculate_failure(len(tests), group9)


def test_seven():
    """
    Testing the SRL model on the test data for Polysemy - Lexical Semantics
    """
    df = pd.read_json("testdata/test7.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:
        sentence = test['sentence']
        golden_labels = test['propbank_golden_labels']
        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)
        output_group_9 = pred.predict_SRL_group9(sentence, constituent)

        total_srl += pred.classify(output_srl, golden_labels)
        total_srl_bert += pred.classify(output_srl_bert, golden_labels)
        total_group_9 += pred.classify(output_group_9, golden_labels)

    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)


def test_eight():
    """
    Testing the SRL model on the test data for spelling errors
    """
    df = pd.read_json("testdata/test8.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:
        sentence = test['error_sentence']
        golden_labels = test['propbank_golden_labels']

        predicate = golden_labels['V']

        try:
            constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)
            output_group_9 = pred.predict_SRL_group9(sentence, constituent)
            score_srl_bert = pred.classify(output_srl_bert, golden_labels)
            score_group_9 = pred.classify(output_group_9, golden_labels)
        except:
            score_srl_bert = 0
            score_group_9 = 0

        try:
            output_srl = pred.predict_SRL(sentence, predicate)
            score_srl = pred.classify(output_srl, golden_labels)
        except:
            score_srl = 0

        total_srl += score_srl
        total_srl_bert += score_srl_bert
        total_group_9 += score_group_9

    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)

def test_nine():
    df = pd.read_json("testdata/test9.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:

        sentence = test['sentence']
        golden_labels = test['propbank_golden_labels']
        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)
        output_group_9 = pred.predict_SRL_group9(sentence, constituent)

        total_srl += pred.classify(output_srl, golden_labels)
        total_srl_bert += pred.classify(output_srl_bert, golden_labels)
        total_group_9 += pred.classify(output_group_9, golden_labels)


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)


if __name__ == "__main__":

    print("######################################")
    print("##  WELCOME TO THE ADVANCED NLP      ##")
    print("##         TAKE HOME EXAM           ##")
    print("######################################")
    print()
    print("This exam has been created by Jonas Kouwenhoven.")
    print("In this Python script, we will be evaluating three Semantic Role Labeling (SRL) models on different tests.")


    print("\nRunning Test One: One verb Sentences")
    srl1, bert1, group91 = test_one()

    # print("\nRunning Test Two: Statement vs. Question")
    # srl2, bert2, group92 = test_two()

    # print("\nRunning Test Three: Active vs. Passive")
    # srl3, bert3, group93 = test_three()

    # print("\nRunning Test Four: Instruments")
    # srl4, bert4, group94 = test_four()

    print("\nRunning Test Five: Context")
    srl5, bert5, group95 = test_five()

    # print("\nRunning Test Six: Synonyms")
    # srl6, bert6, group96 = test_six()

    # print("\nRunning Test Seven: Polysemy")
    # srl7, bert7, group97 = test_seven()

    # print("\nRunning Test Eight: Spelling Errors")
    # srl8, bert8, group98 = test_eight()

    # print("\nRunning Test Nine: Long Distance Dependencies")
    # srl9, bert9, group99 = test_nine()

    # columnnames = ["Test Name", "SRL", "SRL BERT", "SRL Group 9"]

    # testnames = ["One Proposition -> Multiple Realizations", "Statement vs. Question", "Active vs. Passive", "Clefts", "Verb Alternations/Levin's classes", "Synonyms", "Polysemy", "Spelling Errors", "Long Distance Dependencies"]
    # srl = [srl1, srl2, srl3, srl4, srl5, srl6, srl7, srl8, srl9]
    # bert = [bert1, bert2, bert3, bert4, bert5, bert6, bert7, bert8, bert9]
    # group9 = [group91, group92, group93, group94, group95, group96, group97, group98, group99]

    # df = pd.DataFrame(list(zip(testnames, srl, bert, group9)), columns=columnnames)
    # df.to_csv("results.csv", index=False)



    # print("Done")