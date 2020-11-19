import csv
import streamlit as st
from language_model import load_model, decide
import time


model, tokenizer = load_model("./eitherio")

def answer(prediction):
    if prediction > 0.8:
        return "Well, *{choice1}* is a no brainer. Forget about *{choice2}*."
    elif prediction > 0.6:
        return "I would argue in favor of *{choice1}*. Sounds better than *{choice2}*."
    elif prediction > 0.5:
        return " Hard to tell... But if you ask me, I would choose *{choice1}* over *{choice2}* "
    elif prediction > 0.4:
        return " Hard to tell... But if you ask me, I would choose *{choice2}* over *{choice1}*"
    elif prediction > 0.3:
        return "I would argue in favor of *{choice2}*. Sounds better than *{choice1}*."
    else:
        return "Well, *{choice2}* is a no brainer. Forget about *{choice1}*."

def main():
    # feedback = False

    st.title('Hi, I am Pierre Corneille Bot.')
    st.image("pierre_corneille_bot.png", width=150)
    st.write('Let me help you decide on things. You can write two options below, I will decide on one.')

    choice1 = st.text_input("Rather...")
    choice2 = st.text_input("... or...")

    if not(choice1 and choice2):
        pass
    else:
        with st.spinner('Thinking...'):
            prediction = decide([choice1, choice2], model, tokenizer)
        st.write(answer(prediction).format(choice1=choice1, choice2=choice2))
        st.text("Prediction: " + str(prediction[0]))
    
    
    st.subheader('About the name')
    st.write("The name Pierre Corneille Bot comes from French tragedian Pierre Corneille, renowned for his dilemma referred to as \"cornellian\" in which his characters are forced to choose from two options, each having a detrimental effect.")
    st.subheader('About the system')
    st.write("Pierre Corneille Bot uses a quite simple neural network trained on top of an embedding layer frozen to GLoVe word embeddings. The training dataset was scraped from http://either.io/. The neural network had to predict the ratio of people choosing one option over the other from the data of the website.")
    st.subheader('About the author')
    st.write("I am Adam Oudad. You can check my github repository at https://github.com/adamoudad and read my blog at https://adamoudad.github.io/")
        # # st.radio(, ["Yes", "No"], index=-1)
        # if not(feedback):
        #     st.write("Do you agree with me?")
        #     if st.button("Yes"):
        #         with open('user_feedbacks.csv', 'a', encoding="utf-8") as f:
        #             writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        #             writer.writerow([choice1, choice2, prediction, 1])
        #     elif st.button("No"):
        #         with open('user_feedbacks.csv', 'a', encoding="utf-8") as f:
        #             writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        #             writer.writerow([choice1, choice2, prediction, 0])
        #     else:
        #         st.write("I dont know what this is doing.")
        #     feedback = True


if __name__ == '__main__':
    main()
