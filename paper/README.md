# Paper Source

LaTeX source for *A Hybrid GNN-LLM Fusion Architecture for Multi-Modal
Fraud Detection in Ethereum Smart Contracts*.

## Layout

```
paper/
  main.tex            Master document. Uses IEEEtran (conference).
  references.bib      Bibliography in BibTeX format.
  sections/           One .tex file per section, included from main.tex.
    01_introduction.tex
    02_motivation.tex
    03_background.tex
    04_methodology.tex
    05_challenges.tex
    06_prototype.tex
    07_results.tex
    08_conclusion.tex
  figures/            PDF and PNG figures referenced from main.tex.
```

## Build

```
cd paper
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

or with latexmk:

```
latexmk -pdf main.tex
```

## Figures

The figures are regenerable and are not tracked in git. Generate them
before building the paper:

```
python figures/make_figures.py
```

This writes `figures/architecture.pdf` (the architecture block diagram)
and `figures/confusion_matrix.png` (a representative test-split confusion
matrix matching the counts in Section VII). The authoritative confusion
matrix and all dataset plots are produced by the reference implementation
under `../code/`; after running the pipeline you can copy them in with:

```
cp ../code/data/processed/figures/*.png figures/
```
