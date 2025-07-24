import streamlit as st
import Whatsapp_Chat_Analyzer
import Whatsapp_Chat_helper
import matplotlib.pyplot as plt

st.sidebar.title("Whatsapp Chat Analyser")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file:
    bytes_chat_data = uploaded_file.getvalue()
    chat_data = bytes_chat_data.decode("utf-8")
    df = Whatsapp_Chat_Analyzer.preprocess_func(chat_data)
    st.dataframe(df)

    user_list = df["user_name"].unique().tolist()
    #user_list.remove("Group Notification")
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user_name = st.sidebar.selectbox("Select a User", user_list)

    if st.sidebar.button("Show Analysis"):
        col1, col2, col3, col4 = st.columns(4)
        total_messages, total_words, total_media, total_edit_message = Whatsapp_Chat_helper.fetch_info(selected_user_name, df)

        with col1:
            st.header("Total Messages")
            st.title(total_messages)

        with col2:
            st.header("Total words")
            st.title(total_words)

        with col3:
            st.header("Total Media")
            st.title(total_media)

        with col4:
            st.header("Total Edit Messages")
            st.title(total_edit_message)

        if selected_user_name == "Overall":
            st.title("Most busy Users")
            x, new_df = Whatsapp_Chat_helper.most_busy_user(df)
            fig, axis = plt.subplots()
            col5, col6 = st.columns(2)
            with col5:
                axis.bar(x.index, x.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col6:
                st.dataframe(new_df)

        unwanted_df = Whatsapp_Chat_helper.remove_unneeded_rows(df)
        unwanted_df["message"] = unwanted_df["message"].apply(Whatsapp_Chat_helper.remove_stopwords)
        unwanted_df.drop(columns=['index'], inplace=True)

        df_wc = Whatsapp_Chat_helper.create_word_cloud(selected_user_name, unwanted_df)
        fig1, ax1 = plt.subplots()
        ax1.imshow(df_wc)
        st.pyplot(fig1)

        st.dataframe(unwanted_df)

        emoji_df = Whatsapp_Chat_helper.emoji_finder(unwanted_df)
        st.dataframe(emoji_df)
        vader_sentiment_analysis_df = Whatsapp_Chat_helper.sentiment_analysis(unwanted_df)
        st.dataframe(vader_sentiment_analysis_df)
        Hugging_face_sentiment_analysis_df= Whatsapp_Chat_helper.hugging_face(unwanted_df)
        st.dataframe(Hugging_face_sentiment_analysis_df)
        print('completed')
