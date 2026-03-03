# Makefile for EY Challenge 2026 Documentation
# Features:
# - Stable output filename for Git friendliness
# - Clean build directory (dist/)
# - Bilingual support (XeLaTeX/LuaLaTeX compatible)

# Configuration
DOCname = main
OUT_DIR = dist
VERSION_FILE = VERSION.tex
REPORT_NAME = EY_Challenge_2026_Report

# Commands
LATEX = lualatex
FLAGS = -interaction=nonstopmode -output-directory=$(OUT_DIR)

.PHONY: all clean

all: clean
	@mkdir -p $(OUT_DIR)
	@if [ ! -f $(VERSION_FILE) ]; then echo "1.0" > $(VERSION_FILE); fi
	@VERSION=$$(cat $(VERSION_FILE)); \
	echo "Compiling $(DOCname) version $$VERSION..."; \
	$(LATEX) $(FLAGS) $(DOCname).tex; \
	$(LATEX) $(FLAGS) $(DOCname).tex; \
	mv $(OUT_DIR)/$(DOCname).pdf "$(OUT_DIR)/$(REPORT_NAME).pdf"; \
	rm -f $(OUT_DIR)/*.aux $(OUT_DIR)/*.log $(OUT_DIR)/*.toc $(OUT_DIR)/*.out $(OUT_DIR)/*.fls $(OUT_DIR)/*.fdb_latexmk; \
	echo "Build successful! Output: $(OUT_DIR)/$(REPORT_NAME).pdf"

clean:
	@echo "Cleaning up build artifacts..."
	rm -rf $(OUT_DIR)/*
	@echo "Cleaning up root directory artifacts..."
	rm -f *.aux *.log *.toc *.out *.fls *.fdb_latexmk *.synctex.gz
