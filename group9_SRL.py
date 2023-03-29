from bert4srl import predict_group9
# from bert4srl import bert_utils as util




def predict_total(sentence):
        splited_sentence = sentence.split()
        # last word is word with punctuation
        last_word = splited_sentence[-1]
        # remove punctuation from last word
        last_word = last_word[:-1]
        # add last word without punctuation to list
        splited_sentence[-1] = last_word
        # add punctuation to list
        splited_sentence.append(sentence[-1])
        
        
        labels = ["_" for _ in range(len(splited_sentence))]

        labels[-1] = "V"

        result, prediction = predict_group9.predictions([splited_sentence], [labels])
        new_sent = prediction[0][0]
        prediction = prediction[0][1][1:-1]
        return prediction, new_sent


if __name__ == "__main__":
    predict_total("I am a student.")