# Makefile for EY Challenge 2026 Documentation
# Features:
# - Auto-incrementing versioning
# - Clean build directory (dist/)
# - Bilingual support (XeLaTeX/LuaLaTeX compatible)

# Configuration
DOCname = main
OUT_DIR = dist
VERSION_FILE = VERSION.tex

# Commands
LATEX = lualatex
FLAGS = -interaction=nonstopmode -output-directory=$(OUT_DIR)

.PHONY: all clean

all: clean
	@mkdir -p $(OUT_DIR)
	@if [ ! -f $(VERSION_FILE) ]; then echo "0.0" > $(VERSION_FILE); fi
	@awk '{printf "%.1f", $$1 + 0.1}' $(VERSION_FILE) > $(VERSION_FILE).tmp && mv $(VERSION_FILE).tmp $(VERSION_FILE)
	@VERSION=$$(cat $(VERSION_FILE)); \
	echo "Compiling $(DOCname) version $$VERSION..."; \
	$(LATEX) $(FLAGS) $(DOCname).tex; \
	$(LATEX) $(FLAGS) $(DOCname).tex; \
	mv $(OUT_DIR)/$(DOCname).pdf "$(OUT_DIR)/EY_Challenge_2026_Report_v$${VERSION}.pdf"; \
	echo "Build successful! Output: $(OUT_DIR)/EY_Challenge_2026_Report_v$${VERSION}.pdf"

clean:
	@echo "Cleaning up build artifacts..."
	rm -rf $(OUT_DIR)/*
	@echo "Cleaning up root directory artifacts..."
	rm -f *.aux *.log *.toc *.out *.fls *.fdb_latexmk *.synctex.gz
