from bert4srl import predict_group9
import string
# from bert4srl import bert_utils as util




def predict_total(sentence):
        splited_sentence = sentence.split()
        # last word is word with punctuation
        last_word = splited_sentence[-1]
        # remove punctuation from last word
        if last_word[-1] in string.punctuation:
                last_word = last_word[:-1]
                # add punctuation to list
                splited_sentence[-1] = last_word
                splited_sentence.append(sentence[-1])
        else:
                # add last word with existing punctuation to list
                splited_sentence[-1] = last_word
        
        
        labels = ["_" for _ in range(len(splited_sentence))]

        labels[-1] = "V"
        # print(labels, splited_sentence)
        _, pred = predict_group9.predictions([splited_sentence], [labels])
        sentence, prediction, token = pred[0]
        prediction = prediction[1:-1]
        print(prediction, token)

        print("")
        # remove begin token and and token from prediction\
        # prediction
        prediction_new = [value for token, value in zip(token, prediction) if not token.startswith("##")]


        print(len(sentence), len(prediction_new))
        print(sentence, prediction_new)
        return sentence, prediction_new


if __name__ == "__main__":
    predict_total("The cat is pursuing the dog")