import spacy

nlp = spacy.load("en_core_web_md")

doc1 = nlp("I like fast food")
doc2 = nlp("I like pizza")
doc3 = nlp("I like eating healthy")
doc4 = nlp("The story highlights the dichotomy between freedom and control.")
doc5 = nlp("I don't like fast food")

print(doc1.similarity(doc2))
print(doc1.similarity(doc3))
print(doc1.similarity(doc4))
print(doc1.similarity(doc5))

print(doc2[2].vector)
