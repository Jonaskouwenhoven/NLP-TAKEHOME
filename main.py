import pandas as pd
import os
import pandas as pd
from allennlp_models import pretrained
from utils.group9_SRL import predict_total
import re
import string as string_value
# from predict import pred.predict_SRL, pred.predict_SRL_BERT, pred.predict_SRL_group9
import utils.predict as pred
import warnings
import utils.util as util
import utils.classify as classify
warnings.filterwarnings('ignore')

MODEL_NAME_BERT = "logfiles/BERT_SRL/bertSRL"
MODEL_NAME_GROUP_9 = "logfiles/NSRL/NSRL"
MODEL_NAME_SRL = "logfiles/bi-LSTM/bi_LSTM"

MODEL_NAMES = [MODEL_NAME_BERT, MODEL_NAME_SRL, MODEL_NAME_GROUP_9]

def run_predicate_test(test_name, test_data):
    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in test_data:
        sentence = test['original_sentence']
        golden_labels = test['propbank_golden_labels']
        predicate = golden_labels['V']

        try:
            constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)
            output_group_9 = pred.predict_SRL_group9(sentence, constituent)

            score_srl_bert = classify.check_pred(predicate, output_srl_bert, MODEL_NAME_BERT, sentence)
            score_group_9 = classify.check_pred(predicate, output_group_9, MODEL_NAME_GROUP_9, sentence)
        except:

            score_srl_bert = 0
            score_group_9 = 0
            util.log_error(MODEL_NAME_BERT, {"V":"not found"}, {"V": predicate}, sentence, error_type="no_predicate")
            util.log_error(MODEL_NAME_GROUP_9, {"V":"not found"}, {"V": predicate}, sentence, error_type="no_predicate")

        try:
            output_srl = pred.predict_SRL(sentence, predicate)
            score_srl = classify.check_pred(predicate, output_srl, MODEL_NAME_SRL, sentence )

        except:
            score_srl = 0
            util.log_error(MODEL_NAME_SRL, {"V":"not found"}, {"V": predicate}, sentence, error_type="no_predicate")
        
        total_srl += score_srl
        total_srl_bert += score_srl_bert
        total_group_9 += score_group_9

    return pred.calculate_failure(len(test_data), total_srl), pred.calculate_failure(len(test_data), total_srl_bert),  pred.calculate_failure(len(test_data), total_group_9)


def test_one():
    """
    Testing the SRL model on the test data for one verb sentences
    """
    df = pd.read_json("newtestdata/1_one_verb.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")
    return run_predicate_test("One verb sentences", tests)


def test_two():
    """
    Testing the SRL model on the test data for spelling errors
    """
    df = pd.read_json("newtestdata/2_spelling_errors.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")
    return run_predicate_test("Spelling errors", tests)


def test_six():
    """
    Testing the SRL model on the test data for spelling errors
    """
    df = pd.read_json("newtestdata/6_slang.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")
    return run_predicate_test("Slang", tests)  



def test_three():
    """
    Testing SRL capabiltie of handeling, active -> passive
    """
    df = pd.read_json("newtestdata/3_activevsPassive.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    srl, bert, group9 = 0, 0, 0
    for test in tests:
        active = test['active']
        passive = test['passive']

        golden_labels = test['propbank_golden_labels'][0]

        active_gold = golden_labels['Active']
        passive_gold = golden_labels['Passive']

        copy_active_gold = active_gold.copy()
        copy_passive_gold = passive_gold.copy()
        active_predicate = active_gold['V']
        passive_predicate = passive_gold['V']

        constituent, output_active_srl_bert = pred.predict_SRL_BERT(active, active_predicate)
        output_active_srl = pred.predict_SRL(active, active_predicate)
        output_active_group_9 = pred.predict_SRL_group9(active, constituent)
        del copy_active_gold['V']


        constituent, output_passive_srl_bert = pred.predict_SRL_BERT(passive, passive_predicate)
        output_passive_srl = pred.predict_SRL(passive, passive_predicate)
        output_passive_group_9 = pred.predict_SRL_group9(passive, constituent)

        del copy_passive_gold['V']


        bert += (classify.check_arguments(copy_active_gold, output_active_srl_bert, MODEL_NAME_BERT, active) * classify.check_arguments(copy_passive_gold, output_passive_srl_bert, MODEL_NAME_BERT, passive))
        srl += (classify.check_arguments(copy_active_gold, output_active_srl, MODEL_NAME_SRL, active)* classify.check_arguments(copy_passive_gold, output_passive_srl, MODEL_NAME_SRL, passive))
        group9 += (classify.check_arguments(copy_active_gold, output_active_group_9, MODEL_NAME_GROUP_9, active) * classify.check_arguments(copy_passive_gold, output_passive_group_9, MODEL_NAME_GROUP_9, passive))


    return pred.calculate_failure(len(tests), bert), pred.calculate_failure(len(tests), srl),  pred.calculate_failure(len(tests), group9)


def test_four():
    """
    Testing the SRL model on the test data for instruments
    """
    df = pd.read_json("newtestdata/4_instruments.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:
        sentence = test['original_sentence']

        golden_labels = test['propbank_golden_labels']
        instrument = golden_labels['ARG2']
        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)
        output_group_9 = pred.predict_SRL_group9(sentence, constituent)

        total_srl += classify.check_instrument_context(output_srl, instrument, model_name=MODEL_NAME_SRL, original_sentence=sentence)
        total_srl_bert += classify.check_instrument_context(output_srl_bert, instrument, model_name=MODEL_NAME_BERT, original_sentence=sentence)
        total_group_9 += classify.check_instrument_context(output_group_9, instrument, model_name=MODEL_NAME_GROUP_9, original_sentence=sentence)


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)

def test_five():
    """
    Testing the SRL model on the test data for contextual infromation
    """
    df = pd.read_json("newtestdata/5_context.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:
        sentence = test['original_sentence']

        golden_labels = test['propbank_golden_labels']
        manner = golden_labels['ARGM-MNR']

        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)
        output_group_9 = pred.predict_SRL_group9(sentence, constituent)

        total_srl += classify.check_instrument_context(output_srl, manner, 'ARGM-MNR', model_name=MODEL_NAME_SRL, original_sentence=sentence)
        total_srl_bert += classify.check_instrument_context(output_srl_bert, manner, 'ARGM-MNR', model_name=MODEL_NAME_BERT, original_sentence=sentence)
        total_group_9 += classify.check_instrument_context(output_group_9, manner, 'ARGM-MNR', model_name=MODEL_NAME_GROUP_9, original_sentence=sentence)


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)



def test_seven():
    """
    Testing the SRL model on the test data for contextual infromation
    """
    df = pd.read_json("newtestdata/7_agent_patient_long_distance.json")
    tests = df['tests']
    print(f"Number of tests: {len(tests)}")

    total_srl, total_srl_bert, total_group_9 = 0, 0, 0
    for test in tests:
        sentence = test['original_sentence']

        golden_labels = test['propbank_golden_labels']

        predicate = golden_labels['V']
        constituent, output_srl_bert = pred.predict_SRL_BERT(sentence, predicate)

        output_srl = pred.predict_SRL(sentence, predicate)
        output_group_9 = pred.predict_SRL_group9(sentence, constituent)


        del golden_labels['V']

        total_srl += classify.check_long_distance(golden_labels, output_srl, sentence, MODEL_NAME_SRL)
        total_srl_bert += classify.check_long_distance(golden_labels, output_srl_bert, sentence, MODEL_NAME_BERT)
        total_group_9 += classify.check_long_distance(golden_labels, output_group_9, sentence, MODEL_NAME_GROUP_9)


    return pred.calculate_failure(len(tests), total_srl), pred.calculate_failure(len(tests), total_srl_bert), pred.calculate_failure(len(tests), total_group_9)





if __name__ == "__main__":

    print("######################################")
    print("##  WELCOME TO THE ADVANCED NLP      ##")
    print("##         TAKE HOME EXAM           ##")
    print("######################################")
    print()
    print("This exam has been created by Jonas Kouwenhoven.")
    print("In this Python script, we will be evaluating three Semantic Role Labeling (SRL) models on different tests.")

    util.clear_error_files([MODEL_NAME_SRL, MODEL_NAME_BERT, MODEL_NAME_GROUP_9])

    print("\nRunning Test One: One verb Sentences")
    util.start_message(MODEL_NAMES, "One Verb Sentences")
    srl1, bert1, group91 = test_one()


    util.start_message(MODEL_NAMES, "Spelling Errors")
    print("\nRunning Test Two: Spelling Errors")
    srl2, bert2, group92 = test_two()
    # print(f"\n{MODEL_NAME_SRL} failed {srl2} out of 100 tests.")
    # print(f"{MODEL_NAME_BERT} failed {bert2} out of 100 tests.")


    util.start_message(MODEL_NAMES, "Active vs. Passive")
    print("\nRunning Test Three: Active vs. Passive")
    srl3, bert3, group93 = test_three()

    util.start_message(MODEL_NAMES, "Instruments")
    print("\nRunning Test Four: Instruments")
    srl4, bert4, group94 = test_four()

    util.start_message(MODEL_NAMES, "Contexts")
    print("\nRunning Test Five: Context")
    srl5, bert5, group95 = test_five()

    util.start_message(MODEL_NAMES, "Slang")
    print("\nRunning Test Six: Slang")
    srl6, bert6, group96 = test_six()

    util.start_message(MODEL_NAMES, "Long Distance")
    print("\nRunning Test Seven: Long Distance")
    srl7, bert7, group97 = test_seven()


    columnnames = ["Test Name", "SRL", "SRL BERT", "NSRL"]
    testnames = ['One verb', 'Spelling Errors', 'Active vs. Passive', 'Instruments', 'Context', 'Slang', 'Long Distance']
    # testnames = ["1", "2", "3", "4", "5", "6", "7"]
    srl = [srl1, srl2, srl3, srl4, srl5, srl6, srl7]
    bert = [bert1, bert2, bert3, bert4, bert5, bert6, bert7]
    group9 = [group91, group92, group93, group94, group95, group96, group97]

    df = pd.DataFrame(list(zip(testnames, srl, bert, group9)), columns=columnnames)

    # Round the values in the DataFrame to 2 decimal places
    df = df.round(2)

    # Create a LaTeX table and save it to a file
    with open("results.tex", "w") as f:
        f.write(df.to_latex(index=False))

    print("Done")
