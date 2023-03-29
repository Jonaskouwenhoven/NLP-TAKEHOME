from predict import predict_SRL, predict_SRL_BERT, predict_srl_group9

sentence = "I am a student."


print(predict_SRL_BERT(sentence))
print(predict_srl_group9(sentence))

print(predict_SRL("I am a sentence"))


def test_one():

    df = pd.read_json("data/test1.jsonl", lines=True)