#!/bin/python3.6
import logging
import os
import re
from pathlib import Path

import nltk
import treetaggerwrapper
from treetaggerwrapper import TreeTaggerError
from langdetect import detect
from nltk.tokenize.punkt import PunktSentenceTokenizer


NLTK_PUNKT_PATH = os.path.join(Path.home(), "nltk_data", "tokenizers", "punkt")
nltk.data.path.insert(0, NLTK_PUNKT_PATH)
TREETAGGER_LOCATION = os.path.join(Path.home(), "treetagger")

logging.info(f"nltk tokenizers path is set to '{NLTK_PUNKT_PATH}'")
logging.info(f"TREETAGGER_LOCATION is set to '{TREETAGGER_LOCATION}'")

dict_iso_to_full_language_name = {
    # ISO-639-1 language code to the full language name
    # python module iso639 is not used here as the 'el' input
    # returns modern greek (1453-)
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "et": "estonian",
    "fi": "finnish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "fr": "french",
    "it": "italian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
}


def load_sent_tokenizer(language: str):
    """return the nltk sent_tokenizer object
    associated with language
    language should be a string containing the ISO-639-1 language code
    languages.get return the language name associated:
    de-> "german"
    if the language is not in the dict_iso_to_full_language_name, it loads a
    basic tokenizer using
    nltk.tokenize.punkt.PunktSentenceTokenizer
    """

    try:
        tokenizer_name = dict_iso_to_full_language_name[language] + ".pickle"
        sent_tokenizer = nltk.data.load(tokenizer_name)

    except KeyError as e:
        logging.error(
            f"sentence tokenization for language '{language}' not implemented"
        )
        logging.error(f"loading nltk default tokenizer")
        tokenizer_name = "nltk PunktSentenceTokenizer"
        sent_tokenizer = PunktSentenceTokenizer()

    except IndexError as e:
        logging.error(f"cannot load tokenizer, is nltk path right?")
        logging.error(f"nltk path is: {nltk.data.path}")
        logging.error(f"tokenizer '{tokenizer_name}' does not exists?")
        raise e

    logging.info(f"tokenizer '{tokenizer_name}' successfully loaded")
    return sent_tokenizer


def tokenize_into_sentences(text, language) -> list:
    """return a list of tupples
    matching the sentences in the text,
    according to the language parameter
    """
    logging.debug(f"loading sentence tokenizer")
    tokenizer = load_sent_tokenizer(language)
    logging.debug(f"done loading sentence tokenizer")
    logging.info(f"tokenizing text into sentences")
    return [sent for sent in tokenizer.span_tokenize(text)]


class TreeTaggerImproved(treetaggerwrapper.TreeTagger):
    """same class as TreeTagger objects,
    but witht a tag_text_tokens() method that give the token positions
    >>> import utils.nlp_utils as nlp_utils
    >>> tagger = nlp_utils.TreeTaggerImproved(TAGLANG='fr')
    >>> tagger.tag_text_tokens("Elles mangent des pommes.")
        [{'word': 'Elles', 'pos': 'PRO:PER', 'lemma': 'elle', 'start': 0, 'end': 5},
        {'word': 'mangent', 'pos': 'VER:pres', 'lemma': 'manger', 'start': 6,
        'end': 13},
        ...
    """

    def __init__(self, **kwargs):
        logging.debug(f"init TreeTaggerImproved with '{kwargs}'")
        try:
            super().__init__(**kwargs)
        except TreeTaggerError as e:
            logging.error(f"error while initializing the tagger '{e}'")
            logging.error(f"will init from direct TAGPARFILE")
            logging.error(f"The TAGLANG parameter will be set to German")
            tagparfile = f'{TREETAGGER_LOCATION}/lib/{dict_iso_to_full_language_name[kwargs["TAGLANG"]]}.par'  # XXX
            logging.error(f"The TAGPARFILE parameter will be set to '{tagparfile}'")
            kwargs["TAGLANG"] = "de"
            kwargs["TAGPARFILE"] = tagparfile
            lemmatizer = super().__init__(**kwargs)

    def assert_correct_lang_text(self, text, strict=False):
        """if no TAGLANG parameter is used for instanciation,
        the default self.lang is English
        This method assert the self.lang is the same language
        as the text passed in argument"""
        detected_language = detect(text)
        if detected_language != self.lang:
            msg = f"detected language of text is  '{detected_language}', but self.lang is '{self.lang}'. Did you instanciate the tagger with TAGLANG parameter? >>> nlp_utils.TreeTaggerImproved(TAGLANG='fr')"
            if strict:
                raise ValueError(msg)
            logging.error(msg)

    def tag_text_tokens(self, text, check_language=True):
        """return a list of dict containing the tokens:
        [{'word': 'Elles', 'pos': 'PRO:PER', 'lemma': 'elle', 'start': 0, 'end': 5},
        {'word': 'mangent', 'pos': 'VER:pres', 'lemma': 'manger', 'start': 6,
        'end': 13}"""
        if check_language:
            self.assert_correct_lang_text(text)
        tags = self.tag_text(
            text,
            notagurl=True,
            notagemail=True,
            notagip=True,
            notagdns=True,
            nosgmlsplit=True,
        )
        ls_tags = treetaggerwrapper.make_tags(tags)
        ls_tokens = []
        last_pos = 0
        num_errors = 0
        for i, tag in enumerate(ls_tags):
            # print(i, tag, f"last_pos: {last_pos}")
            word = tag[0]
            if word == "...":
                # by default TT changes '…' to '...'
                # this causes misleading characters count
                # note that the token can also be '...'
                # we check that the token has not been replaced:
                if re.match(r"\s*…", text[last_pos:]):
                    # the token has been replaced from '…' to '...'
                    logging.debug("remplacing ... by …")
                    word = "…"
            # counting the number of spaces after last token
            num_spaces_after_last_token = re.match("\s*", text[last_pos:]).span()[1]
            try:
                # the new token position is the sum of the last position,
                # and the index of the text containing the word.
                # upper limit in the search method in the text
                # with num_spaces_after_last_token, otherwise,
                # the index search could go very far in the text
                token_position = last_pos + text[
                    last_pos : last_pos + num_spaces_after_last_token + len(word)
                ].index(word)
            except Exception as e:
                logging.warning(
                    f"cannot find 1st substring (word is: '{word}'). Checking another position ..."
                )
                # checking if the token is an Acronym as
                # Acronyms like U.S.A. are systematically written
                # with a final dot, even if it is missing in original file. See
                # https://treetaggerwrapper.readthedocs.io/en/latest/#other-things-done-by-this-module
                token_position = None
                try:
                    token_position = last_pos + text[last_pos:].index(word.strip("."))
                    word = word.strip(".")
                except Exception as e2:
                    logging.critical(f"cannot find {tag} even without trailing dot")
                    raise e2
                if token_position is None:
                    # the token position could be equal to 0
                    # so checking for the booleaness can raise Error
                    logging.critical(f"ignoring {tag}")
                    logging.critical(f"\ttoken position: {token_position}")
                    logging.critical(f"\tlast_position {last_pos}")
                    logging.critical(f"\t{e}")
                    num_errors += 1
                    # continue
                    raise e
                logging.warning(f"-> found corresponding substring")
            if token_position - last_pos > 100:
                logging.warning(
                    f"token position is too far. Last pos is {last_pos} new pos is {token_position}"
                )
            if type(tag) == treetaggerwrapper.Tag:
                pos = tag[1]
                lemma = tag[2]
            elif type(tag) == treetaggerwrapper.NotTag:
                pos = ""
                lemma = word
            token = {
                "word": word,
                "pos": pos,
                "lemma": lemma,
                "start": token_position,
                "end": token_position + len(word),
            }
            ls_tokens.append(token)
            last_pos = token["end"]
        if num_errors:
            logging.error(f"{num_errors} on {i} tokens")
        # last assertion
        for token in ls_tokens:
            assert (
                token["word"] == text[token["start"] : token["end"]]
            ), "the token position doesnt match in the text"
        return ls_tokens

    def get_text_lemmatized(self, text, check_language=True):
        "return a lemmatized version of the text"
        if check_language:
            self.assert_correct_lang_text(text)
        tokens = self.tag_text_tokens(text, check_language=check_language)
        return " ".join([word["lemma"] for word in tokens])


if __name__ == "__main__":
    pass
