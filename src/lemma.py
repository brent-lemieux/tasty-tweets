import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


# if not 'nlp' in locals():
#     print "Loading English Module..."
#     nlp = spacy.load('en')
#
# STOPLIST = set(list(ENGLISH_STOP_WORDS) + ["n't", "'s'", "'m'"])
#
# def lemmatize_string(doc, stop_words):
#     doc = unicode(doc.translate(None, punctuation))
#     doc = nlp(doc)
#     tokens = [token.lemma_ for token in doc]
#     return ' '.join(w for w in tokens if w not in stop_words)
