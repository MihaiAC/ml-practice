import spacy

nlp = spacy.blank("en")
doc = nlp("I have cat")

cat_hash = nlp.vocab.strings["cat"]
print(cat_hash)

cat_string = nlp.vocab.strings[cat_hash]
print(cat_string)

doc = nlp("David Bowie is a PERSON")

person_hash = nlp.vocab.strings["PERSON"]
print(person_hash)

person_string = nlp.vocab.strings[person_hash]
print(person_string)
