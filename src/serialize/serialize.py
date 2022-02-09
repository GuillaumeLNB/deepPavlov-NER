"""
"""
import csv
import inspect
import json
import html
import logging
import os
import sys
from datetime import datetime

from tqdm import tqdm
from yattag import Doc

__location__ = os.path.join(
    os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe()))
)


class Serializer:
    """This class provides methods to export the results
    of nlp tasks to different formats (HTML, tsv, ttl)"""

    def __init__(self, text: str, json_data: list, safe_check=True):
        """init the serializer

        Args:
            text (str): the text string
            json_data (list[dict]): the annotations linked to the text


        """

        self.text = text
        # sorting data by start character
        self.data = sorted(json_data, key=lambda k: k["start"])
        if safe_check:
            self.check_data_is_well_formated()

    def __len__(self):
        return len(self.data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(f"exc_type: {exc_type}", file=sys.stderr)
            print(f"exc_value: {exc_value}", file=sys.stderr)
            print(f"exc_traceback: {exc_traceback}", file=sys.stderr)

    def check_data_is_well_formated(self):
        """log errors if the annotation is wrongly formated"""
        for annotation in self.data:
            if annotation["start"] > annotation["end"]:
                logging.critical(
                    f"annotation '''{annotation}''' annotation['start'] is more than annotation['end']"
                )
            if (
                not annotation["text"]
                == self.text[annotation["start"] : annotation["end"]]
            ):
                logging.critical(
                    f"annotation['text'] doesn't match the part of the text with annotation['start']:annotation['end']"
                )

    def export_to_format(self, out_file: str, format_: str):
        """wrapper to export the annotations to the
        format given in argument"""
        if format_ == "tsv":
            self.to_tsv(out_file)
        elif format_ == "json":
            self.to_json(out_file)
        elif format_ == "html":
            self.to_html(out_file)
        else:
            self.to_linked_data(out_file, format_)

    def to_tsv(self, out_file: str):
        """export the annotations to tsv format"""
        logging.info(f"exporting the annotations to tsv {out_file}")

        with open(out_file, "w", encoding="UTF-8") as csv_file:
            fieldnames = ["text", "annotation", "start", "end"]
            writer = csv.DictWriter(
                csv_file, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
        logging.info(f"outfile:\t{out_file}")

    def to_json(self, out_file: str):
        """export the annotations to json format"""
        logging.info(f"exporting the annotations to json {out_file}")
        with open(out_file, "w", encoding="UTF-8") as f:
            json.dump(self.data, f, indent=4)
        logging.info(f"outfile:\t{out_file}")

    def to_temporary_json(self, out_file: str):
        """export the annotations to a specific json format that contains:

            - the annotation (LOC, ORG, RT, ...)
            - the text of the annotation (Ringo Starr, Obama, ...)
            - all of the prefix string
            - the start prefix ONLY (for testing purposes)

        :param out_file: the output file
        :type out_file: str
        """

        logging.info(f"exporting the annotations to a temporary json file: {out_file}")

        temporary_data = []
        nb_char_suffix = 100  # for performance reason 100
        # suffix length is the best compromise

        for anno in tqdm(self.data):

            # the javascript count should only take
            # the first occurence of the prefix
            # the prefix is the string of minimum length
            # the we can use to find the annotation
            # however, in order to desambiguate,
            # if the prefix is present more than one time in the
            # document, it will create a longer prefix

            for i in range(nb_char_suffix, anno["start"], 50):
                # print(i)
                prefix_start = anno["start"] - i
                prefix_string = self.text[prefix_start : anno["start"]]
                # if self.text.count(prefix_string) > 1:
                if occurrences(self.text, prefix_string) > 1:
                    # print("OCCURENCE")
                    # using occurences as count does not return overlaping
                    # strings
                    continue
                break
            else:
                prefix_start = 0
                prefix_string = self.text[: anno["start"]]

            assert prefix_start + len(prefix_string) == anno["start"]
            assert prefix_start == self.text.index(prefix_string)

            suffix = self.text[
                anno["end"] : min(len(self.text), anno["end"] + nb_char_suffix)
            ]
            if anno["start"]:
                assert prefix_string, anno
            # node will compute the annotation start and end using
            # its way of counting characters
            # it will ouput the prefixes and suffixes as well
            temporary_data.append(
                {
                    "annotation": anno["annotation"],
                    "text": anno["text"],
                    "start": anno["start"],  # only used for testing
                    # "prefix": self.text[: anno["start"]],
                    "prefix": prefix_string,
                    "suffix": suffix,
                }
            )
        with open(out_file, "w", encoding="UTF-8") as f:
            json.dump(temporary_data, f, indent=4)
        logging.info(f"outfile:\t{out_file}")

    def to_html(self, out_file: str):
        """export the annotation in a html page

        if annotations are overlapping,
        it will skip the last annotation

        """
        logging.info(f"exporting the annotations to html: '{out_file}'")

        doc, tag, text = Doc().tagtext()
        doc.asis("<!DOCTYPE html>")
        with tag("html"):
            with tag("head"):
                doc.stag("meta", charset="UTF-8")
                with tag("style", ("type", "text/css")):
                    doc.asis(get_style_sheet())
            logging.info("creating the HTML...")
            with tag("body"):
                with tag("h3"):
                    text(f"{datetime.now()}"[:16])
                    doc.asis(f"<br>")
                    text("_" * 50)
                    doc.asis(f"<br>")
                    # text(f"detected language: {self.lang}")
                    doc.asis("<br>")
                with tag("table", ("style", "width:85%"), ("align", "center")):
                    with tag("tr"):
                        with tag("td", ("width", "70%")):
                            with tag("div", ("class", "td_text")):
                                html_text = self.text
                                # escaping html characters
                                # as a regular html.escape() demands more work
                                # on the string length
                                html_text = html_text.replace(
                                    "<", "ᐸ"
                                )  # 1438	 CANADIAN SYLLABICS PA
                                html_text = html_text.replace(
                                    ">", "ᐳ"
                                )  # 1433	 CANADIAN SYLLABICS PO
                                html_text = html_text.replace("&", "﹠")
                                # new_text=re.sub('"', "''", new_text)
                                ls_ent_type = []
                                for ent in sorted(
                                    self.data, key=lambda k: k["start"], reverse=True
                                ):
                                    ls_ent_type.append(ent["annotation"])
                                    if "title" not in ent:
                                        ent["title"] = ent["annotation"]

                                    html_text = (
                                        html_text[: ent["start"]]
                                        + '<span class="'
                                        + ent["annotation"].lower()
                                        + '" title="'
                                        + html.escape(ent["title"])
                                        + " &#10;"
                                        + ent.get("reason", "")
                                        + '">'
                                        + ent["text"]
                                        + "</span>"
                                        + html_text[ent["end"] :]
                                    )
                                html_text = html_text.replace("\n", "<br>\n")
                                doc.asis(html_text)
                        with tag("td", ("width", "30%"), ("class", "legend")):
                            for type_ in sorted(
                                set([ann["annotation"] for ann in self.data])
                            ):
                                with tag("span", ("class", type_.lower())):
                                    text(type_)
                                    doc.asis("<br>")

        with open(out_file, "w", encoding="UTF-8") as file:
            print(doc.getvalue(), file=file)
        logging.info(f"outfile:\t{out_file}")


def get_style_sheet():
    "return the text contained in the css file"

    style_sheet_path = os.path.join(__location__, "..", "css", "style.css")

    with open(style_sheet_path) as f:
        css = f.read()
    return css


def occurrences(text, sub):
    count = start = 0
    while True:
        start = text.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count
