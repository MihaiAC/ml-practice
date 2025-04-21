import spacy

nlp = spacy.load("en_core_web_md")
print(f"Pipe names: \n{nlp.pipe_names}")
print(f"Pipeline: \n{nlp.pipeline}")
