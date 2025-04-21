import spacy

nlp = spacy.load("en_core_web_sm")
text = "Upcoming iPhone X release data leaked as Apple reveals pre-orders"
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)

iphoneX_ent = doc[1:3]
print("Missing entity:", iphoneX_ent.text)
