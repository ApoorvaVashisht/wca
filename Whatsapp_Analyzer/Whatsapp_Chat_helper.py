import re
from wordcloud import WordCloud
import Whatsapp_Chat_Analyzer
import pandas as pd
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import torch

emoji_pattern = re.compile(
    "["                   
    "\U0001F600-\U0001F64F"  
    "\U0001F300-\U0001F5FF"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F1E0-\U0001F1FF"  
    "\u2600-\u26FF"          
    "\u2700-\u27BF"          
    "]+"
)

def fetch_info(selected_use_name, df):
    words = []
    media_num = 0
    edit_message = 0
    if selected_use_name != "Overall":
        new_df = df[df["user_name"] == selected_use_name]
    else:
        new_df = df
    for message_content in new_df["message"]:
        if "<Media omitted>" in message_content:
            media_num += 1
        if "<This message was edited>" in message_content:
            edit_message += 1
        words += message_content.split(" ")
    return new_df.shape[0], len(words), media_num, edit_message


def most_busy_user(df):
    x = df["user_name"].value_counts()
    y = ((x / df.shape[0]) * 100).reset_index().rename(columns={'user_name': 'User', 'count': 'Chat %'})
    return x, y


def create_word_cloud(selected_user, df):
    if selected_user != "Overall":
        new_df = df[df["user_name"] == selected_user]
    else:
        new_df = df
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(new_df['message'].str.cat(sep=" "))
    return df_wc


def remove_unneeded_rows(df):
    temp_df = df[df["user_name"] != "Group Notification"]
    temp_df = temp_df[temp_df["message"] != "<Media omitted>\n"]
    return temp_df.reset_index()


def remove_stopwords(text):
    hin_eng_stopwords = Whatsapp_Chat_Analyzer.read_hin_eng_stopwords()
    temp_word_list = []
    text_split = text.split(" ")
    for text_word in text_split:
        if text_word.lower() not in hin_eng_stopwords:
            temp_word_list.append(text_word)
    return " ".join(temp_word_list)


def emoji_finder(df):
    emoji_list = []
    for message_text in df["message"]:
        emoji_list.extend(emoji_pattern.findall(message_text))
    emoji_content = Counter(emoji_list)
    emoji_df = pd.DataFrame({'emoji': emoji_content.keys(), 'Count': emoji_content.values()})
    return emoji_df

def sentiment_analysis(df):
    print("Starting sentiment analysis")
    #nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    for message in df['message']:
        print(sid.polarity_scores(message))

    df[['negative', 'neutral', 'positive', 'compound']] = df['message'].apply(sid.polarity_scores).apply(pd.Series)
    print(df)
    df['sentiment_type'] = df['compound'].apply(extract_sentiment)
    print(df)
    return df
def extract_sentiment(compound_val):
    if compound_val < -0.05:
        return 'Negative'
    elif compound_val > 0.05:
        return 'Positive'
    else:
        return 'Neutral'

def hugging_face(df):
    sentiment = pipeline("sentiment-analysis")
    df['hf_result'] = df['message'].apply(lambda x: sentiment(x)[0])
    print(df)
    df['hf_label'] = df['hf_result'].apply(lambda r: r['label'])
    print(df)
    df['hf_score'] = df['hf_result'].apply(lambda r: r['score'])
    print(df)
    return df