import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Union

from torchfusion.core.constants import DataKeys
from torchfusion.core.data.text_utils.utilities import get_bbox_center
from torchfusion.core.utilities.logging import get_logger


class TesseractOCRReader:
    def __init__(self, ocr_file_path: Union[str, Path]):
        from pathlib import Path

        import bs4

        logger = get_logger()
        try:
            ocr_file = Path(ocr_file_path)
            if ocr_file.exists() and ocr_file.stat().st_size > 0:
                with open(ocr_file, "r", encoding="utf-8") as f:
                    xml_input = eval(f.read())
                self._soup = bs4.BeautifulSoup(xml_input, features="xml")
            else:
                logger.warning(f"Cannot read file: {ocr_file}.")
        except Exception as e:
            logger.exception(
                f"Exception raised while reading ocr data from file {ocr_file}: {e}"
            )

    @cached_property
    def pages(self):
        return self._soup.findAll("div", {"class": "ocr_page"})

    @cached_property
    def blocks(self):
        return self._soup.findAll("div", {"class": "ocr_carea"})

    @cached_property
    def words(self):
        return self._soup.findAll("span", {"class": "ocrx_word"})

    def pars_per_block(self, block):
        return block.findAll("p", {"class": "ocr_par"})

    def lines_per_par(self, par):
        return par.findAll("span", {"class": "ocr_line"})

    def words_per_line(self, line):
        return line.findAll("span", {"class": "ocrx_word"})

    @cached_property
    def image_size(self):
        image_size_str = self.pages[0]["title"].split("; bbox")[1]
        w, h = map(int, image_size_str[4 : image_size_str.find(";")].split())
        return (w, h)

    def parse(self):
        words = []
        word_bboxes = []
        word_angles = []
        w, h = self.image_size
        for word in self.words:
            title = word["title"]
            conf = int(title[title.find(";") + 10 :])
            if word.text.strip() == "" or conf < 50:
                continue

            # get text angle from line title
            textangle = 0
            parent_title = word.parent["title"]
            if "textangle" in parent_title:
                textangle = int(parent_title.split("textangle")[1][1:3])

            x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())
            words.append(word.text.strip())
            word_bboxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            word_angles.append(textangle)
        return words, word_bboxes, word_angles


class TesseractOCRGraphReader(TesseractOCRReader):
    @dataclass
    class NodesInfo:
        first = None
        last = None

    def __init__(self, ocr_file_path: Union[str, Path]):
        super().__init__(ocr_file_path=ocr_file_path)

        import networkx as nx

        self._global_node = 0
        self._graph = nx.Graph()

    def add_node(self, node_attr):
        node = self._global_node
        self._graph.add_node(node, **node_attr)
        self._global_node += 1
        return node

    def add_edge(self, from_id, to_id):
        import numpy as np

        # get euclidean distnace between nodes in the image
        from_center = np.array(
            get_bbox_center(self._graph.nodes[from_id][DataKeys.WORD_BBOXES])
        )
        to_center = np.array(
            get_bbox_center(self._graph.nodes[to_id][DataKeys.WORD_BBOXES])
        )
        dist = np.linalg.norm(from_center - to_center)

        # add distance between nodes as weight of edges
        # weights are same as cost incurred, more weight = bad
        self._graph.add_edge(from_id, to_id, weight=dist)
        self._graph.add_edge(to_id, from_id, weight=dist)

    def parse(self):
        prev_block_nodes = self.NodesInfo()
        w, h = self.image_size
        for block in self.blocks:
            prev_par_nodes = self.NodesInfo()
            pars_per_block = self.pars_per_block(block)
            for par_idx, par in enumerate(pars_per_block):
                prev_line_nodes = self.NodesInfo()
                lines_per_par = self.lines_per_par(par)
                for line_idx, line in enumerate(lines_per_par):
                    words_per_line = self.words_per_line(line)
                    prev_word_node = None
                    for word_idx, word in enumerate(words_per_line):
                        # find word confidence score from word title
                        title = word["title"]
                        conf = int(title[title.find(";") + 10 :])
                        if word.text.strip() == "" or conf < 50:
                            continue

                        # get word text
                        text = word.text.strip()

                        # find bounding box coordinates from word title
                        x1, y1, x2, y2 = map(int, title[5 : title.find(";")].split())

                        # get text angle from word title
                        angle = 0
                        parent_title = word.parent["title"]
                        if "textangle" in parent_title:
                            angle = int(parent_title.split("textangle")[1][1:3])

                        # add word as node to graph
                        node = self.add_node(
                            {
                                DataKeys.WORDS: text,
                                DataKeys.WORD_BBOXES: [x1 / w, y1 / h, x2 / w, y2 / h],
                                DataKeys.WORD_ANGLES: angle,
                            }
                        )

                        # attach word to prev node in line
                        if prev_word_node is not None:
                            self.add_edge(node, prev_word_node)

                        # update first line node and first paragraph node
                        if prev_word_node is None:
                            if (
                                prev_line_nodes.first is not None
                            ):  # attach first node of line to first node of previous line
                                self.add_edge(node, prev_line_nodes.first)

                            if (
                                prev_line_nodes.last is not None
                            ):  # attach first node of line to last node of previous line
                                self.add_edge(node, prev_line_nodes.last)

                            if line_idx == 0:
                                if (
                                    prev_par_nodes.first is not None
                                ):  # attach first node of paragraph to first node of prev paragraph
                                    self.add_edge(node, prev_par_nodes.first)

                                if (
                                    prev_par_nodes.last is not None
                                ):  # attach first node of paragraph to last node of prev paragraph
                                    self.add_edge(node, prev_par_nodes.last)

                                if par_idx == 0:
                                    if (
                                        prev_block_nodes.first is not None
                                    ):  # attach first node of block to first node of prev block
                                        self.add_edge(node, prev_block_nodes.first)

                                    if (
                                        prev_block_nodes.last is not None
                                    ):  # attach first node of block to last node of prev block
                                        self.add_edge(node, prev_block_nodes.last)

                                    prev_block_nodes.first = node

                                prev_par_nodes.first = node

                            prev_line_nodes.first = node

                        prev_word_node = node

                    # update last line node and last paragraph node
                    prev_line_nodes.last = prev_word_node
                prev_par_nodes.last = prev_line_nodes.last
            prev_block_nodes.last = prev_par_nodes.last
        return self._graph
