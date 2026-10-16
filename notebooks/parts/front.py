"""Front matter: title page, abstract, how to read, table of contents."""

from notebooks.notebook_builder import code, md

FACTS = '''
# Run the notebook from the repository root. F holds the numbers quoted in the running text, read from the repository's own files: a sentence such as
# "{{=F.n_train:,}} molecules" is filled in from F when the notebook is built. The show_* helpers put figures, tables and equations into a cell's output.
from pathlib import Path

from notebooks.facts import load
from notebooks.notebook_builder import show_figure, show_markdown, show_math, show_table

ROOT = Path.cwd()
F = load()
'''

CELLS = [
    md("""
<div align="center">

# CYPMode

### Predicting how a molecule switches off a liver enzyme, explained from zero

A complete account of the CYPMode project: the biology and chemistry of the problem, the data, the statistics, the machine learning, the quantum-chemical features and the
structure prediction behind an entry to the OpenADMET CYP blind challenge, every experiment that was run (the ones that failed as carefully as the ones that worked), and the
code that ran them.

**Muhammad Talaat**<br>Faculty of Pharmacy, Ain Shams University

</div>
""", titlepage=True),
    code(FACTS, hide_input=True),
    md("""
## Abstract

Many medicines are removed from the body by a small number of liver enzymes. A molecule that switches one of these enzymes off, and keeps it off, can make other medicines build
up to harmful levels. This notebook explains, starting from nothing, how a computer can be taught to recognise such molecules and to predict how they sit inside the enzyme, and
it documents one complete attempt: the CYPMode project, an entry to the OpenADMET CYP blind challenge.

Two tasks are solved. The first is classification: for each of {{=F.n_test}} molecules whose measurements are hidden, say whether it switches off the enzyme CYP3A4, and whether it
switches off the enzyme CYP2D6, in the slow and lasting way called time-dependent inhibition. The second is structure prediction: for 20 molecules, predict where every atom sits
when the molecule is bound inside CYP3A4.

Because the answers stay hidden until the challenge closes, the central difficulty is to judge honestly how good a model is. The project therefore evaluates every model by
nested cross-validation, in which no choice the model makes (which members to trust, how to blend them, where to put the yes/no threshold) is tested on the data it was chosen on,
and it judges every proposed improvement by a rule written down before its result existed. {{=F.gates}} such rules were judged: {{=F.gates_not_met}} ended "not met" and {{=F.gates_met}}
ended "met". The final classification model blends {{=F.n_models}} models that read {{=F.n_columns:,}} numbers per molecule. In nested cross-validation it reaches an area under the ROC curve
(a score that is 0.5 for guessing and 1 for a perfect ranking, defined in section 4.10) of {{=F.auc['CYP3A4']:.3f}} for CYP3A4 and {{=F.auc['CYP2D6']:.3f}} for CYP2D6. These are
estimates made on the training data. The challenge score itself is not known until the challenge closes. The submitted structures number {{=F.poses}}, and {{=F.pose_failures}} of them fail the
challenge's physical-validity checks.

The notebook is also a record of what did not work. Every idea that was tried is described with the rule it had to pass, its numbers and the reason it did or did not help, and the
code of the whole project is mapped to the sections that explain it (Appendix D). All numbers in the text are produced by code from the files of the repository when the notebook is
built; none is typed by hand.
"""),
    md("""
## How to read this notebook

- **No background is assumed.** A reader with no chemistry, biology, statistics or machine learning can start at the first page. Each term is defined in bold where it first appears,
  and each equation is built from named pieces, step by step, with a worked example computed in a code cell.
- **Parts build on one another.** Part 0 gives the whole picture on a few pages and points to the section that explains each piece. Parts I to IV teach the biology, the data, the
  chemistry-as-numbers and the machine learning that the later parts use. A reader who already knows a topic can skip to the part that needs it; the table of contents below lists every section.
- **Code cells are real.** Where a code cell appears it runs against the files of this repository when the notebook is built, and a cell that fails stops the build. The cells show
  how a number was obtained; they do not reproduce a multi-hour computation, which is read from the saved results of that computation instead.
- **Figures are the project's own.** Each is drawn by code in `docs/figures/` from the same files, shown here on a dark panel with a caption that says what it shows and where its
  data come from.
- **Failures are part of the text.** An experiment that did not pass its gate is explained as carefully as one that did. Where a part of the code does less than its name suggests,
  a paragraph headed "On what actually ran" says so.
- **Every idea has an origin.** Where a method or formula comes from other people's work, the text says whose and cites it (Appendix C). Where an idea is this project's own, the text
  says that too and shows the evidence.
- **Conventions.** Names of files, modules and functions are written `like this`. A **bold** word is defined at that point. Symbols are introduced where they are first needed and
  collected in Appendix B. Section numbers refer to sections of this notebook.
"""),
    md("<!-- contents -->", placeholder="contents", generated=True),
]
