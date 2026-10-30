# Map of the project

Four views of one project, and one page that draws them so that they can be explored. Every page is written by `python -m cypmode.audit.project_map` from the files of the repository, and `python -m cypmode.audit.project_map --check` fails when a page differs from them, so a page cannot drift from the code.

| page | what it shows | size |
| --- | --- | --- |
| [The code](code.md) | the packages and modules, what each module does and what it imports | 172 modules |
| [The queue](queue.md) | the jobs, what each waits for and the chain that rebuilds and checks the project | 171 jobs |
| [The flow of the data](dataflow.md) | from the inputs through the jobs and the logs to the reports and the submission files | 5 stages |
| [The evidence](evidence.md) | the kinds of idea that were tried and the ledger that counts the verdicts | one ledger |
| [The interactive map](explorer.html) | all four views as one page that can be explored: a graph of the modules, a grid of the jobs, a search box, and for each one what it is connected to | one file |

The interactive map is one file with its data inside: download it and open it in a browser (GitHub shows an HTML file as text), or open it on the Pages site. What the map does not show: the content of any input or result, and what happens while a job runs. It names files and counts them, and it reads no log.
