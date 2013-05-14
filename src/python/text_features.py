import pandas as pd
import nltk 
import re
from features import readBusiness, readReview, combineTestTrain
from sklearn.feature_extraction import DictVectorizer


def pos_tag_text(text):
    words = nltk.word_tokenize(text)
    text_length = float(len(words))
    pos_tags  = nltk.pos_tag(words)
    counts = {}
    for word, pos in pos_tags:
        count = counts.setdefault(pos, 0)
        counts[pos] = count + 1.0 / text_length
    print(counts)
    return(counts)

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

def ARI(nchars, nwords, nsents):
    return 4.71 * (nchars/ nwords) + 0.5 * (nwords / nsents) - 21.43

def CLI(nchars, nwords, nsents):
    return 0.0588 * (nchars / nwords) - 0.296 * (nsents / nwords) - 15.8



# TODO normalize (by textlength or tfidf)
def getTextFeatures(review, business):
    data = review.reset_index().merge(business, left_on = "business_id", right_index = True, how = "left").set_index("review_id")
    textFeatures = pd.DataFrame(index = data.index)
    data["text"] = data.text.map(lambda x: x if type(x) == type(str()) else "")
    print("Getting POS features")
    vec = DictVectorizer()
    pos_dicts = data.text.map(pos_tag_text).values
    pos_features = vec.fit_transform(pos_dicts).toarray()
    pos_dataFrame = pd.DataFrame(pos_features, index = textFeatures.index, columns = vec.get_feature_names())
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
    #textFeatures = textFeatures.apply(lambda x: (float(x) / float(textlengths)), axis = 0)
    textFeatures = textFeatures.fillna(0.0)
    textFeatures["textlength"] = textlengths
    print("readability indices")
    textFeatures["ARI_Readability"] = ARI(textFeatures["textlength"], textFeatures["number_of_words"], textFeatures["number_of_sentences"])
    textFeatures["CLI_Readability"] = CLI(textFeatures["textlength"], textFeatures["number_of_words"], textFeatures["number_of_sentences"])
    return textFeatures.combine_first(pos_dataFrame)
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