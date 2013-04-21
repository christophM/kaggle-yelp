import pandas as pd
import nltk 
import re
from features import readBusiness, readReview, combineTestTrain

#def countOccurences(text, words):
    ## tokenize by word

def countSymbols(text, symbol):
    return text.count(symbol)


def lengthOfText(text):
    return len(text)
    
def numberOfSentences(text):
     sent = nltk.sent_tokenize(text)
     count = [1 for i in sent].count(1)
     return(count)


def numberOfWords(text):
     word = nltk.word_tokenize(text)
     return([1 for i in word].count(1))

def countParagraphs(text):
    return text.count("\n\n")

def countNewlines(text):
    return text.count("\n")

def countOccurencesOfBizName(text, biz_name):
    return text.lower().count(biz_name.lower())


def countCapitalLetters(text):
    return len(re.findall("[A-Z]", text))


# TODO normalize (by textlength or tfidf)
def getTextFeatures(review, business):
    data = review.reset_index().merge(business, left_on = "business_id", right_index = True, how = "left").set_index("review_id")
    textFeatures = pd.DataFrame(index = data.index)
    data["text"] = data.text.map(lambda x: x if type(x) == type(str()) else "")
    print("counting")
    textFeatures["count_capital_letters"] = data.text.map(countCapitalLetters)
    textFeatures["count_text_biz_name"] = data.apply(lambda x: countOccurencesOfBizName(x["text"], x["name"]), axis = 1, raw = True)
    textFeatures["count_newlines"] = data.text.map(countNewlines)
    textFeatures["count_paragraphs"] = data.text.map(countParagraphs)
    textFeatures["count_exmarks"] = data.text.map(lambda x: countSymbols(x, "!"))
    print("tokenizing")
    textFeatures["number_of_words"] = data.text.map(numberOfWords)
    textFeatures["number_of_sentences"] = data.text.map(numberOfSentences)
    textlengths = data.text.map(lengthOfText)
    #textFeatures.apply(lambda x: (float(x) / float(textlengths)), axis = 0)
    textFeatures.fillna(0.0)
    textFeatures["length_of_text"] = textlengths
    return textFeatures
    
def main():
    test_path = "./data/test/"
    train_path = "./data/train/"
    test = readReview(test_path + "yelp_test_set_review.csv")
    train = readReview(train_path + "yelp_academic_dataset_review.csv")
    train = train[["text", "business_id"]]
    test = test[["text", "business_id"]]
    businesses_train = readBusiness(train_path + "yelp_academic_dataset_business.csv")
    businesses_test = readBusiness(test_path + "yelp_test_set_business.csv")
    business = combineTestTrain(businesses_train, businesses_test)
    trainTextFeatures = getTextFeatures(train, business)
    testTextFeatures = getTextFeatures(test, business)
    trainTextFeatures.to_csv(train_path + "features-text-train.csv")
    testTextFeatures.to_csv(test_path + "features-text-test.csv")

    
    
if __name__ == "__main__":
    main()