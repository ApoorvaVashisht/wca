import re
import pandas as pd


def preprocess_func(chat_data):
    user_name, user_message = [], []
    #msg_time_pattern = r"[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}[\s\u202f\u00a0]?(?:AM|PM|am|pm)\s-\s"
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}[\s\u202f\u00a0]?(?:AM|PM|am|pm)\s-\s'
    pattern_timestamp = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}[\s\u202f\u00a0]?(?:AM|PM|am|pm))'
    chat_msg_data = (re.split(pattern, chat_data))[1:]
    chat_time_data = re.findall(pattern_timestamp, chat_data)
    df = pd.DataFrame({'message_time': chat_time_data, 'messages': chat_msg_data})
    df["message_time"] = pd.to_datetime(df["message_time"], format='%d/%m/%y, %I:%M %p')
    message_split_pattern = r":\s"
    for message_content in df["messages"]:
        message_content_split = re.split(message_split_pattern, message_content)
        if len(message_content_split) > 1:
            user_name.append(message_content_split[0])
            user_message.append(message_content_split[1])
        else:
            user_name.append("Group Notification")
            user_message.append(message_content_split[0])
    df['user_name'] = user_name
    df['message'] = user_message
    df["Year"] = df["message_time"].dt.year
    df["Month"] = df["message_time"].dt.month_name()
    df["Date"] = df["message_time"].dt.day
    df["Hour"] = df["message_time"].dt.hour
    df["Minute"] = df["message_time"].dt.minute
    df.drop(columns=['messages'], inplace=True)
    df.drop(columns=["message_time"], inplace=True)

    return df


def read_hin_eng_stopwords():
    file_name = "C:\\Users\\Apoorva Vashisth\\PycharmProjects\\pythonProject\\Whatsapp_Analyzer\\stop_hinglish.txt"
    with open(file_name, 'r') as hin_eng_file:
        file_data = hin_eng_file.read()
    file_data_split = file_data.split("\n")
    return file_data_split
