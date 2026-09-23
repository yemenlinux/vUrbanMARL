# Minimal makefile for Sphinx documentation

export LC_ALL ?= C.UTF-8
export LANG ?= C.UTF-8

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" html $(SPHINXOPTS) $(O)
	@touch html/.nojekyll

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
