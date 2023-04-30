import utils.util as util

def check_long_distance(gold_data, predicted_data, original_sentence, model_name):
    gold_patient = gold_data.get("ARG0", "").lower()
    predicted_patient = predicted_data.get("ARG0", "").lower()
    if gold_patient not in predicted_patient:
        util.log_error(model_name, predicted_data, gold_data, original_sentence, error_type="patient")
        return 0

    gold_agent = gold_data.get("ARG1", "").lower()
    predicted_agent = predicted_data.get("ARG1", "").lower()
    if gold_agent not in predicted_agent:
        util.log_error(model_name, predicted_data, gold_data, original_sentence, error_type="agent")
        return 0
    util.log_success(model_name, predicted_data, gold_data, original_sentence)
    return 1


def check_pred(gold_pred, output, model_name, sentence):
    if 'V' in output:
        srl_pred = output['V']
        if srl_pred == gold_pred:
            util.log_success(model_name, output, {"V": gold_pred}, sentence)
            return 1
        else:
            util.log_error(model_name, output, {"V": gold_pred}, sentence, error_type="predicate_mismatch")
            return 0
    else:
        util.log_error(model_name, output, {"V": gold_pred}, sentence, error_type="no_predicate")
        return 0



def check_arguments(gold, predicted, model_name, original_sentence):
    for key in gold.keys():
        if key in predicted.keys():
            if gold[key].lower() == predicted[key].lower():
                continue
            else:
                util.log_error(model_name, predicted, gold, original_sentence, error_type="argument_mismatch")
                return 0
        else:
            util.log_error(model_name, predicted, gold, original_sentence, error_type="argument_missing")
            return 0
    util.log_success(model_name, predicted, gold, original_sentence)
    return 1



def check_instrument_context(output, argument, arg_label='ARG2', model_name=None, original_sentence=None):
    if arg_label in output:
        if argument in output[arg_label]:
            util.log_success(model_name, output, {arg_label: argument}, original_sentence)
            return 1
        else:
            util.log_error(model_name, output, {arg_label: argument}, original_sentence, error_type=f"{arg_label}_mismatch")
            return 0
    else:
        util.log_error(model_name, output, {arg_label: argument}, original_sentence, error_type=f"{arg_label}_missing")
        return 0
