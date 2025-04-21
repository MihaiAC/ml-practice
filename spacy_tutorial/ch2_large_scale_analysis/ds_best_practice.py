import spacy

# Convert tokens to the string as late as possible

nlp = spacy.load("en_core_web_sm")
doc = nlp("Berlin looks like a nice city")

for token in doc:
    # Check if the current token is a proper noun
    if token.pos_ == "PROPN":
        # Check if the next token is a verb
        if token.i + 1 < len(doc) and doc[token.i + 1].pos_ == "VERB":
            print("Found proper noun before a verb:", token.text)
