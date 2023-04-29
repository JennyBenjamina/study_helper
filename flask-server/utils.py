import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
import re
import cohere
from sklearn.metrics.pairwise import cosine_similarity

co = cohere.Client('PKwpHpAfrm6yzOJc9StFMkWrYj1NUvfTrVtLxznG')


def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # stem each word
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

def word_frequency(tokens):
    word_frequencies = {}
    stop_words = stopwords.words('english')
    for word in tokens:
        if word.lower() not in stop_words:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

def text_summarizer(text):
    tokenized = tokenize(text)
    split_text = text.split()
    # r = re.compile("[a-zA-Z0-9]+")
    # cleaned_text = list(filter(r.match, tokenized))
    bag = bag_of_words(tokenized, split_text)
    print(bag)

    c_text = re.sub(r"[^\w]", ' ', text) # cleaned text
    f = open("clean_transcribed.txt", "w", encoding="utf-8")
    f.write(c_text)
    f.close()

    f2 = open("dirty_transcribed.txt", "w", encoding="utf-8")
    f2.write(text)
    f2.close()

    return tokenized
    

#############################################################################

def generate_prompt(message):
    return """Generate 2 practice questions and answers on the subject {}. 
    Make sure the questions starts with a number followed by a period. 
    Make sure the answer starts with A followed by a colon.""".format(message)


########################################


def embed_text(texts):
  """
  Turns a piece of text into embeddings
  Arguments:
    text(str): the text to be turned into embeddings
  Returns:
    embedding(list): the embeddings
  """
  # Embed text by calling the Embed endpoint
  output = co.embed(
                model="large",
                texts=texts)
  embedding = output.embeddings

  return embedding

def squeeze_if_one(arr):
    # Check if all dimensions are 1
    if all(dim == 1 for dim in arr.shape):
        # Find the first non-1 dimension
        non_one_dim = next((i for i, dim in enumerate(arr.shape) if dim != 1), None)
        if non_one_dim is not None:
            # Squeeze all but the first non-1 dimension
            return np.squeeze(arr, axis=tuple(range(non_one_dim + 1, len(arr.shape))))
    return arr

def get_similarity(target, candidates):
    """
    Computes the similarity between a target text and a list of other texts
    Arguments:
    target(list[float]): the target text
    candidates(list[list[float]]): a list of other texts, or candidates
    Returns:
    sim(list[tuple]): candidate IDs and the similarity scores
    """
    # Turn list into array
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target),axis=0)

    # Calculate cosine similarity
    sim = cosine_similarity(target,candidates)
    sim = squeeze_if_one(sim)
    # sim = np.squeeze(sim)
    # sim = sim.tolist()
    # Sort by descending order in similarity
    sim = list(enumerate(sim))
    sim = sorted(sim, key=lambda x:x[1], reverse=True)
    
    # Return similarity scores
    return sim


