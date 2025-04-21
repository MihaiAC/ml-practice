import spacy
from spacy.tokens import Token

nlp = spacy.blank("en")

Token.set_extension("is_country", default=False)

doc = nlp("I live in Spain.")
doc[3]._.is_country = True

print([(token.text, token._.is_country) for token in doc])
