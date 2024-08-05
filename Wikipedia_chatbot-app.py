import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

# Download required NLTK data files
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def lemma_me(sent):
    sentence_tokens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sentence_tokens)

    sentence_lemmas = []
    for token, pos_tag in zip(sentence_tokens, pos_tags):
        if pos_tag[1][0].lower() in ['n', 'v', 'a', 'r']:
            lemma = lemmatizer.lemmatize(token, pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)
        else:
            # Also lemmatize non-Noun, Verb, Adjective, or Adverb tokens
            lemma = lemmatizer.lemmatize(token)
            sentence_lemmas.append(lemma)

    return sentence_lemmas


def process(text, question):
    sentence_tokens = nltk.sent_tokenize(text)
    sentence_tokens.append(question)

    tv = TfidfVectorizer(tokenizer=lemma_me)
    tf = tv.fit_transform(sentence_tokens)
    values = cosine_similarity(tf[-1], tf)
    index = values.argsort()[0][-2]
    values_flat = values.flatten()
    values_flat.sort()
    coeff = values_flat[-2]
    if coeff > 0.2:
        return sentence_tokens[index]


# Streamlit App
st.set_page_config(page_title='Wikipedia-ChatBot', layout='wide')
st.title('Wikipedia-ChatBot')

# Sidebar for topic selection
with st.sidebar:
    topic_selection = st.text_input("For what topic you want to ask questions:", "MachineLearning")

if topic_selection:
    try:
        with st.spinner('Loading content from Wikipedia...'):
            text = wikipedia.page(topic_selection).content
            st.success(f"Loaded content for the topic: {topic_selection}")

        question = st.text_input("Hi, what do you want to know?")

        if question:
            with st.spinner('Processing your question...'):
                output = process(text, question)
                if output:
                    st.markdown("### Answer:")
                    st.write(output)
                else:
                    st.markdown("### Answer:")
                    st.write("I don't know.")
    except wikipedia.exceptions.PageError:
        st.error("The requested Wikipedia page was not found.")
    except wikipedia.exceptions.DisambiguationError as e:
        st.error("The topic is ambiguous. Possible options are:")
        st.write(e.options)
