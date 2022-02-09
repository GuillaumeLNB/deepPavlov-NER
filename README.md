# deepPavlov-NER
Python deepPavlov for NER (easy to get the entities spans and viz...)


Run the scripts in a virtual environment

### deepPavlov's installation

See http://docs.deeppavlov.ai/en/master/intro/installation.html for deepPavlov's intallation
(don't forget to upgrade pip in the virtual environment!)

### Install deepPavlov's models
`python -m deeppavlov install ner_ontonotes_bert_torch`
`python -m deeppavlov install squad_bert`
`python install.py`

### Install requirements
`python -m pip install -r requirements.txt`
download spacy model:
`python -m spacy download xx_ent_wiki_sm`

# Demo
```
import sys

sys.path.insert(0, "src")

from ner.ner import NETagger
from serialize.serialize import Serializer

text = """
L’Ingénieux Hidalgo Don Quichotte de la Manche ou L'Ingénieux Noble Don Quichotte de la Manche (titre original en espagnol El ingenioso hidalgo don Quixote de la Mancha ; en espagnol moderne : El ingenioso hidalgo don Quijote de la Mancha) est un roman écrit par Miguel de Cervantes et publié à Madrid en deux parties, la première en 1605 puis la seconde en 1615.

À la fois roman médiéval — un roman de chevalerie — et roman de l'époque moderne alors naissante, le livre est une parodie des mœurs médiévales et de l'idéal chevaleresque, ainsi qu'une critique des structures sociales d'une société espagnole rigide et vécue comme absurde. Don Quichotte est un jalon important de l'histoire littéraire et les interprétations qu'on en donne sont multiples : pur comique, satire sociale, analyse politique. Il est considéré comme l'un des romans les plus importants de la littérature mondiale et comme le premier roman moderne.

Le personnage, Alonso Quichano, est à l'origine de l'archétype du Don Quichotte, personnage généreux et idéaliste qui se pose en redresseur de torts.

"""

ner = NETagger(text)
entities = ner.predict()
for entity in entities:
    print(entity)

# {'annotation': 'WORK_OF_ART', 'text': 'L’Ingénieux Hidalgo Don Quichotte de la Manche', 'start': 1, 'end': 47}
# {'annotation': 'WORK_OF_ART', 'text': "L'Ingénieux Noble Don Quichotte de la Manche", 'start': 51, 'end': 95}
# {'annotation': 'NORP', 'text': 'espagnol', 'start': 115, 'end': 123}
# {'annotation': 'WORK_OF_ART', 'text': 'El ingenioso hidalgo don Quixote de la Mancha', 'start': 124, 'end': 169}
# {'annotation': 'NORP', 'text': 'espagnol', 'start': 175, 'end': 183}
# {'annotation': 'WORK_OF_ART', 'text': 'El ingenioso hidalgo don Quijote de la Mancha', 'start': 194, 'end': 239}
# {'annotation': 'PERSON', 'text': 'Miguel de Cervantes', 'start': 264, 'end': 283}
# {'annotation': 'GPE', 'text': 'Madrid', 'start': 296, 'end': 302}
# {'annotation': 'CARDINAL', 'text': 'deux', 'start': 306, 'end': 310}
# ...

with Serializer(text, entities) as s:
    s.to_tsv("entities.tsv")
    s.to_html("entities.html")
    s.to_json("entities.json")


```