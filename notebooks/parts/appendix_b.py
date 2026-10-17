"""Appendix B: symbols, abbreviations and the glossary, all generated from the notebook itself (its text and notebooks/parts/glossary.py), so that the appendix cannot disagree with the text."""

from notebooks.notebook_builder import code, md

SYMBOL_INDEX = '''
# The index of symbols, made from the text of the notebook itself. Every piece of inline mathematics in the running text (the maths between dollar signs, not the display equations) that is short enough to be a symbol or a small expression is
# collected, with the sections in which it appears. A symbol is named in the section where it is used, as the builder requires, so the index says where to look; it does not say what the symbol means, because the same letter can mean different things in different sections.
import re

from notebooks.notebook_builder import DISPLAY, FENCE, INLINE, section_of_cells
from notebooks.notebook_cells import ALL_CELLS

section_labels = section_of_cells(ALL_CELLS)
symbol_sections = {}
for label, cell in zip(section_labels, ALL_CELLS):
    if cell["cell_type"] != "markdown":
        continue
    for found in INLINE.finditer(DISPLAY.sub(" ", FENCE.sub("", cell["source"]))):
        expression = " ".join(found.group(1).split())
        if len(expression) <= 16 and re.search(r"[A-Za-z]", expression) and not re.search(r"[=|]| ", expression):
            symbol_sections.setdefault(expression, [])
            if label not in symbol_sections[expression]:
                symbol_sections[expression].append(label)
symbol_rows = [{"symbol": f"`{expression}`", "sections": ", ".join(places[:8]) + (f" and {len(places) - 8} more" if len(places) > 8 else "")} for expression, places in sorted(symbol_sections.items(), key=lambda kv: kv[0].lower().lstrip("\\\\"))]
symbol_table = pd.DataFrame(symbol_rows)
show_table(symbol_table, floatfmt=".0f")
n_symbols = len(symbol_table)
letter_sections = symbol_sections.get("A", [])
'''

ABBREVIATIONS = '''
# The abbreviations, read from the glossary: a term that is written mostly in capitals (at least two capital letters, at least 60 percent of its letters, no spaces) is an abbreviation, and its meaning is the glossary's.
import re

from notebooks.parts.glossary import GLOSSARY

abbreviations = sorted((t for t in GLOSSARY if re.fullmatch(r"[A-Za-z0-9+/-]+", t) and sum(ch.isupper() for ch in t) >= 2 and sum(ch.isupper() for ch in t) / sum(ch.isalpha() for ch in t) >= 0.6), key=str.lower)
abbreviation_table = pd.DataFrame({"abbreviation": abbreviations, "meaning": [GLOSSARY[t] for t in abbreviations]})
show_table(abbreviation_table, floatfmt=".0f")
n_abbreviations = len(abbreviation_table)
'''

GLOSSARY_TABLE = '''
# The glossary: every term that the notebook defines in bold where it first uses it, in alphabetical order, with the definition of notebooks/parts/glossary.py. The builder fails when a term is used before its definition.
glossary_terms = sorted(GLOSSARY, key=str.lower)
glossary_table = pd.DataFrame({"term": glossary_terms, "meaning": [GLOSSARY[t] for t in glossary_terms]})
show_table(glossary_table, floatfmt=".0f")
n_terms = len(glossary_table)
'''

CELLS = [
    md("""
# Appendix B. Symbols, abbreviations and glossary

This appendix is made by code from the notebook and from the glossary file, so that it is always the appendix of the text beside it. B.1 is an index of the mathematical symbols: where each one is used. B.2 lists the abbreviations. B.3 is the glossary of every term that the notebook defines.

## B.1 Symbols

> **Field:** Mathematics · Documentation

The notebook defines a symbol in the section where it uses it, and the builder checks that every symbol of an equation is named in the text of its section before the equation. It does not keep a single list of meanings, because a single list would be wrong: the same letter is used for different things in different parts of science, and the text follows each field's own convention. The letter $A$ is the adjacency matrix of a molecule in 2.1 and the accuracy of a submitted file in 2.10. What the reader needs from an appendix is therefore the place to look, and the next cell makes that: for every short piece of inline mathematics in the text, the sections in which it appears.
"""),
    code(SYMBOL_INDEX),
    md("""
The index has {{=n_symbols}} entries. The letter $A$ appears in the sections {{=', '.join(letter_sections)}}; to read what it means, go to the first of them and then to the others, as the example of this paragraph does.

## B.2 Abbreviations

> **Field:** Mathematics · Documentation

An abbreviation is a term written in capital letters; its meaning is the glossary's, so that the two cannot differ.
"""),
    code(ABBREVIATIONS),
    md("""
There are {{=n_abbreviations}} abbreviations in the table.

## B.3 Glossary

> **Field:** Mathematics · Documentation

Every term that the notebook defines is listed once, alphabetically, with the definition it was given where it first appeared. The definitions are the short form; the section that introduces a term explains it in full, with its reason and its use.
"""),
    code(GLOSSARY_TABLE),
    md("""
The glossary holds {{=n_terms}} terms.
"""),
]
