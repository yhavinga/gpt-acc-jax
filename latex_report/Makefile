.PHONY: all figures pdf clean

all: figures pdf

figures:
	python create_figures.py

pdf: report.tex
	pdflatex -interaction=nonstopmode report.tex
	pdflatex -interaction=nonstopmode report.tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg

distclean: clean
	rm -f report.pdf
