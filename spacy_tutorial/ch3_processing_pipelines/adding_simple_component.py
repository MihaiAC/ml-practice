import spacy
from spacy.language import Language

nlp = spacy.load("en_core_web_sm")


@Language.component("custom_component")
def custom_component_function(doc):
    print("Doc length:", len(doc))
    return doc


nlp.add_pipe("custom_component", first=True)
print("Pipeline:", nlp.pipe_names)

doc = nlp("Hello world!")
