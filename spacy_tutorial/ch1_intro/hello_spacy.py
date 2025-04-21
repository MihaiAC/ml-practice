import spacy

# Blank English nlp object.
nlp = spacy.blank("en")

# Process a string
doc = nlp("Hello spacy!")

for token in doc:
    print(token.text)
    print(token.i)
    print(token.is_alpha)
