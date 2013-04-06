import pandas as pd
import re

#def countOccurences(text, words):
    ## tokenize by word

def countSymbols(text, symbol):
    return text.count(symbol)


def lengthOfText(text):
    return len(text)

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
    textFeatures["count_capital_letters"] = data.text.map(countCapitalLetters)
    textFeatures["count_text_biz_name"] = data.apply(lambda x: countOccurencesOfBizName(x["text"], x["name"]), axis = 1, raw = True)
    textFeatures["count_newlines"] = data.text.map(countNewlines)
    textFeatures["count_paragraphs"] = data.text.map(countParagraphs)
    textFeatures["count_exmarks"] = data.text.map(lambda x: countSymbols(x, "!"))
    textlengths = data.text.map(lengthOfText)
    #textFeatures.apply(lambda x: (float(x) / textlengths), axis = 0)
    textFeatures.fillna(0.0)
    textFeatures["length_of_text"] = textlengths
    return textFeatures