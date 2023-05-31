#!/bin/python3.6
"""
"""
import inspect
import html
import logging
import os
import re
import sys

# assert sys.version_info[:2] == (3, 6), "works only on python3.6"

import spacy
from deeppavlov import configs, build_model
from langdetect import detect
from tqdm import tqdm
from unidecode import unidecode


__location__ = os.path.join(
    os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe()))
)
sys.path.insert(0, os.path.join(__location__, "..", "nlp_utils"))
from nlp_utils import tokenize_into_sentences

# spacy is used only for tokenization (token level)
SPACY_MODEL = "xx_ent_wiki_sm"
nlp_small = spacy.load(SPACY_MODEL, disable=["ner", "parser", "tagger"])
# setting the maximum number of tokens the pipeline can support
# (safe if ner, parser and tagger are disabled)
nlp_small.max_length = 1_000_000_000

# List of named entities to skip
LIST_ENT_TO_SKIP = [
    # "CARDINAL",
    # "LANGUAGE",
    # "MONEY",
    # "NORP",
    # "ORDINAL",
    # "PERCENT",
    # "PRODUCT",
    # "QUANTITY",
]

# replace the GPE tag to the LOC
GPE_to_LOC = False


class NETagger:
    """wrapper to perform NER

    It uses the deepPavlov build_model() fonctionnality
    and train models with
        ner_ontonotes_bert_mult: the BERT embeddings for multilingual (ie: not English)
        ner_ontonotes_bert     : the BERT embeddings for English

    example:
    >>> ner_model = NETagger()                # will train the model with the ner_ontonotes_bert_mult
    >>> ner_model.new_text(open('textfile').read()) # adding the text
    >>> ner_model.tag_entities()              # tag the entities
    >>> ner_model.serialize(format_='jsonld') # return the jsonld list of entities
    >>> ner_model.to_html('htmlfile.html')    # output the text with hilighted entities

    note that some entities type will be ignored: those contained in LIST_ENT_TO_SKIP
    (CARDINAL, ORDINAL, ...)

    """

    def __init__(
        self,
        text: str,
        language: str = None,
        unescape_html=True,
        default_non_ent="O",
        train_model=True,
    ):
        # logging.debug(f"__init__ NETagger train_model:{train_model} with {config}")
        self.unescape_html = unescape_html
        self._default_non_ent = default_non_ent
        self.new_text(text, language)

        if train_model:
            self._train_model()

    def __str__(self):
        str_ = f"""NETagger object language is '{self.language}'\n"""
        return str_

    # __repr__ = __str__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            logging.error(f"exc_type: {exc_type}")
            logging.error(f"exc_value: {exc_value}")
            logging.error(f"exc_traceback: {exc_traceback}")

    def _train_model(self):
        """train and store the BERT model to self.ner_model

        BERT model used:
            * ner_ontonotes_bert_mult of other language than English
            * ner_ontonotes_bert for English

        """

        if self.language == "en":
            # training the model with ner_ontonotes_bert
            logging.info(f"loading the model with 'ner_ontonotes_bert'")
            # obsolete
            self.ner_model = build_model(configs.ner.ner_ontonotes_bert)
            # self.ner_model = build_model(configs.ner.ner_ontonotes_bert_torch)

        # elif self.language == "ru":
        # the ner_rus_bert model is
        # not working on my pc as the model is too heavy
        # does not work well as the model only contains 4 entity types (not WOA)
        # training the model with ner_rus_bert
        # logging.info(f"training the model with 'ner_rus_bert'")
        # self.ner_model = build_model(configs.ner.ner_rus_bert)
        else:
            #  training the model with ner_ontonotes_bert_mult
            logging.info(f"loading the model with 'ner_ontonotes_bert_mult'")
            # obsolete
            self.ner_model = build_model(configs.ner.ner_ontonotes_bert_mult)
            # self.ner_model = build_model(configs.ner.ner_ontonotes_bert_torch)
        logging.info("done loading the model")

    def new_text(self, text: str, language: str = None):
        """replace the old text with a new one"""
        logging.debug(f"replacing the text with a new one")
        assert isinstance(
            text, str
        ), f"text type should be a string, not a {type(text)}"
        assert len(text), "text is empty"
        if len(text) > 100000:
            logging.warning(
                f"text is long ({len(text)} characters), performing NER on it could cause warnings"
            )
        if self.unescape_html:
            logging.info("unescaping the html entities")
        self.text = html.unescape(text) if self.unescape_html else text
        self.text = html.unescape(
            text
        )  # unescaping the html entites to get better tokenization

        self.text = text
        if not language:
            language = detect(text).split("-")[0]
        self.language = language

    def predict(self, length=250):
        """
        tokenize, perform NER on the self.text attribute, and update self.lis_entities

        BERT models cannot perform on more than 250 tokens. Tried to modify this in the
        deeppavlov settings, and couldn't make it work.

        The workaround proposed is: tokenizing the text into sentences,
        then tokenizing the sentence and run the bert on chunks of theses tokens.
        Note that a sliding window approach could have been proposed but requires
        too much time.

        TOKENIZING THE SENTENCE works best with nltk's sent_tokenize compared to
        spacy and sentence_splitter, because those 2 modules take the \\n as sentence boundaries.

        TOKENIZING THE TEXT INTO TOKENS works better with spacy
        """
        if not hasattr(self, "ner_model"):
            self._train_model()
        if not hasattr(self, "text"):
            raise ValueError("no text to tag")

        ls_tokens, labels = [], []

        # tokenizing into sentences and getting their span
        last_sentence_index = 0
        sentences = tokenize_into_sentences(self.text, self.language)

        logging.info("tagging the entities")

        for sent_start, sent_end in tqdm(sentences):
            sentence = self.text[sent_start:sent_end]
            if last_sentence_index != sent_start:
                # The tokenization of sentences does not keep spaces
                # we need to reconstruct the missing tokens from the text
                # we label those empty tokens as 'O' (no entity)
                ls_tokens.append(
                    {
                        "token": self.text[last_sentence_index:sent_start],
                        "position": (last_sentence_index, sent_start),
                        "trailing_whitespace": "",
                    }
                )
                labels.append(self._default_non_ent)
            last_sentence_index = sent_end

            # tokenizing tokens using the SPACY_MODEL
            tokens = [
                {
                    "token": tok.text,
                    "position": (tok.idx + sent_start, tok.idx + len(tok) + sent_start),
                    "trailing_whitespace": tok.whitespace_,  # Trailing space character if present.
                }
                for tok in nlp_small(sentence)
            ]

            # checking no mistake has been made while computing the token positions
            for tok in tokens:
                assert (
                    tok["token"] == self.text[tok["position"][0] : tok["position"][1]]
                )

            # getting chunks of tokens and performing NER on them
            for tokseq_number, tok_seq in enumerate(chunks(tokens, length), start=1):
                # small_ls_tokens = [tok["token"] for tok in tok_seq]
                if tokseq_number > 1:
                    logging.warning(
                        f"SENTENCE TOO LONG ---"
                        + unidecode(sentence.replace("\n", "\\n"))[:100]
                        + "... (CONVERTED TO ASCII)"
                    )
                    # logging.warning(f"TOKENS ARE ---{tokens[:10]} ... ---")
                try:
                    res = self.ner_model([[tok["token"] for tok in tok_seq]])
                    # raise RuntimeError('raising the error for testing')
                except RuntimeError as e:
                    # if a RuntimeError occurs it can be caused
                    # by a huge proportion of non letter tokens
                    # eg: ['A&lt;^ft.i-', 'j', '-', 'j^^', '\n\n', '/4*.&gt;-&lt;U-', 'rf', '-', 'U', ',', '\n\n'
                    # will ignore this sequence
                    logging.error(e)
                    logging.error(
                        f"list of tokens that ner is performed is: "
                        + unidecode(str(tok_seq))
                    )
                    logging.error(
                        f"will label those tokens as 'not named entities' using '{self._default_non_ent}'"
                    )
                    labels += [self._default_non_ent for _ in tok_seq]
                else:
                    labels += [anno for anno in res[1][0]]
                ls_tokens += tok_seq

            assert len(ls_tokens) == len(
                labels
            ), f"{len(ls_tokens)}\t{len(labels)}, {ls_tokens}, {labels}"
        # self.tokens = ls_tokens
        # self.anno = labels
        tokens = ls_tokens

        assert len(tokens) == len(
            labels
        ), f"{len(tokens)}\t{len(labels)}, {tokens}, {labels}"

        # updating the entities
        entities = []
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # skipping the entity if the entity is in LIST_ENT_TO_SKIP
            if label.split("-")[-1] in LIST_ENT_TO_SKIP:
                continue
            token["label"] = label
            if label == "O":
                continue
            if label.startswith("B-"):
                entities.append(
                    {
                        "annotation": label.split("-")[1],
                        "text": token["token"],
                        "start": token["position"][0],
                        "end": token["position"][1],
                        "trailing_whitespace": token["trailing_whitespace"],
                    }
                )
            elif label.startswith("I-"):
                try:
                    # checking the list of tokens follows the BIO entities rules
                    assert label.split("-")[1] == entities[-1]["annotation"]
                except AssertionError as e:
                    # if a token is labelled as I-something and the previous token is
                    # labelled as  B-somethingElse
                    # it is extremely uncommun, but might occur on certain texts
                    # eg: hugo.notredame.en.txt
                    # it might happen between B-CARDINAL and I-MONEY
                    logging.warning(f"{e}")
                    logging.warning(
                        f"new token has an entity label starting by I- label is:{label}'"
                    )
                    logging.warning(
                        f"but previous token is labelled as: {entities[-1]['annotation']}'"
                    )
                    logging.warning(f"token number is {i}")
                    logging.warning(f"current token is {token}")
                    logging.warning(f"previous token is: {entities[-1]}")
                    logging.warning("will update the token with current label")
                    entities.append(
                        {
                            "annotation": label.split("-")[1],
                            "text": token["token"],
                            "start": token["position"][0],
                            "end": token["position"][1],
                            "trailing_whitespace": token["trailing_whitespace"],
                        }
                    )
                else:
                    entities[-1].update(
                        {
                            "end": token["position"][1],
                            "trailing_whitespace": token["trailing_whitespace"],
                            "text": entities[-1]["text"]
                            + entities[-1]["trailing_whitespace"]
                            + token["token"],
                        }
                    )
            else:
                raise ValueError(f"Incorrect label :{label}")

        # merging the GPE to LOC:
        if GPE_to_LOC:
            for ent in entities:
                if ent["annotation"] == "GPE":
                    ent["annotation"] = "LOC"

        # dropping trailing_whitespace key
        for ent in entities:
            del ent["trailing_whitespace"]

        # stripping the entities as some entities can start with spaces
        for ent in entities:
            # match at the start of the string
            match = re.match(r"\s+", ent["text"])
            if match:
                logging.warning(
                    unidecode(
                        f"entities (unidecoded) starts with whitespace ('{ent}' - {match}) -> updating it"
                    )
                )
                # ent contains starting white space
                # shifting the start to the right
                ent["start"] += match.end()
                ent["text"] = ent["text"][match.end() :]
            # checking the end of the string
            match = re.search(r"\s+$", ent["text"])
            if match:
                logging.warning(
                    unidecode(
                        f"entities (unidecoded) ends with whitespace ('{ent}' - {match}) -> updating it"
                    )
                )
                # truncating the end
                match_length = len(match[0])
                ent["end"] -= match_length
                ent["text"] = ent["text"][: match.start()]
            assert (
                self.text[ent["start"] : ent["end"]] == ent["text"]
            ), f"error when updating an annotation starting or ending with spaces {ent}"
        return entities


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    return


if __name__ == "__main__":
    main()
else:
    logging.info(f"will skip these entitites: {LIST_ENT_TO_SKIP}")
