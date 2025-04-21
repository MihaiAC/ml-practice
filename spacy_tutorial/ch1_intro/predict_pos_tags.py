import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("She ate a pizza.")

# Predict pos tags and syntactic dependencies.
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

# Predicting name entities.
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.label_, spacy.explain(ent.label_))
