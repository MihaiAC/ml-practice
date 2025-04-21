import spacy
from spacy.tokens import Doc

nlp = spacy.blank("en")


def get_has_number(doc):
    return any(token.like_num for token in doc)


Doc.set_extension("has_number", getter=get_has_number)

# Process the text and check the custom has_number attribute
doc = nlp("The museum closed for five years in 2012.")
print("has_number:", doc._.has_number)
