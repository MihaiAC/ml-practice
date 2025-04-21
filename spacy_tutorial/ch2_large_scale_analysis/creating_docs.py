import spacy
from spacy.tokens import Doc, Span

nlp = spacy.blank("en")

words = ["spaCy", "is", "huh", "!"]
spaces = [True, True, False, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

words = ["I", "like", "David", "Glowie"]
spaces = [True, True, True, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)

span = Span(doc, 2, 4, label="PERSON")
print(span.text, span.label_)
