import spacy
from spacy.tokens import Token

nlp = spacy.blank("en")


def get_reversed(token):
    return token.text[::-1]


Token.set_extension("reversed", getter=get_reversed)

doc = nlp("All generalizations are false, including this one.")
for token in doc:
    print("reversed:", token._.reversed)
