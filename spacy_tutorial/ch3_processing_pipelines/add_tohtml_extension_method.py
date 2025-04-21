import spacy
from spacy.tokens import Span

nlp = spacy.blank("en")


def to_html(span, tag):
    return f"<{tag}>{span.text}</{tag}>"


Span.set_extension("to_html", method=to_html)

doc = nlp("Hello world, this is a sentence.")
span = doc[0:2]
print(span._.to_html("strong"))
