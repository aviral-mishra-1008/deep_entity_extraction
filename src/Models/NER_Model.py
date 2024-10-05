'''
Install dependency : pip install spacy
Load the training data from the pkl file
'''

import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
from training_data import load_train_data

TRAIN_DATA = load_train_data()

nlp = spacy.blank("en")
db = DocBin()
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is not None:  
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("./train.spacy")
ner = nlp.add_pipe("ner")

for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

# Train the model
optimizer = nlp.begin_training()
for i in range(100):  # Increase the number of iterations
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer)

