ROOTDIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
OUTDIR := $(abspath $(ROOTDIR)/../docs/paul/)
SRCDIR := $(abspath $(ROOTDIR))
QMDFILES := $(shell find $(SRCDIR) -type f -name '*.qmd')
HTMLFILES := $(patsubst $(SRCDIR)%,$(OUTDIR)%,$(QMDFILES:.qmd=.html))

#QMDWITHCODE := $(shell grep -rl '```{python}' $(QMDFILES))
#NOTEBOOKS := $(patsubst $(SRCDIR)%,$(OUTDIR)%,$(QMDWITHCODE:.qmd=.ipynb))


html: $(HTMLFILES)

all: html

$(OUTDIR)/%.html: $(SRCDIR)/%.qmd $(SRCDIR)/_quarto.yml $(SRCDIR)/styles.css
	quarto render $<

deploy: all
	cd $(OUTDIR); git add *; git commit -a -m "Automated updates to html slides."; git push origin master


test:
	echo $(HTMLFILES)
	echo $(QMDFILES)
