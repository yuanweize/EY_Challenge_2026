# Makefile for EY Challenge 2026 Documentation
# Features:
# - Auto-incrementing versioning
# - Clean build directory (dist/)
# - Bilingual support (XeLaTeX/LuaLaTeX compatible)

# Configuration
DOCname = main
OUT_DIR = dist
VERSION_FILE = VERSION

# Commands
LATEX = lualatex
FLAGS = -interaction=nonstopmode -output-directory=$(OUT_DIR)

.PHONY: all clean version

all: version
	@mkdir -p $(OUT_DIR)
	@echo "Compiling $(DOCname) version $(VERSION)..."
	$(LATEX) $(FLAGS) $(DOCname).tex
	$(LATEX) $(FLAGS) $(DOCname).tex
	@mv $(OUT_DIR)/$(DOCname).pdf $(OUT_DIR)/Project_Documentation_v$(VERSION).pdf
	@echo "Build successful! Output: $(OUT_DIR)/Project_Documentation_v$(VERSION).pdf"

version:
	@if [ ! -f $(VERSION_FILE) ]; then echo 0.0 > $(VERSION_FILE); fi
	@awk 'BEGIN { FS = "." } { printf "%.1f", $$1 + 0.1 }' $(VERSION_FILE) > $(VERSION_FILE).tmp && mv $(VERSION_FILE).tmp $(VERSION_FILE)
	$(eval VERSION := $(shell cat $(VERSION_FILE)))

clean:
	@echo "Cleaning up build artifacts..."
	rm -rf $(OUT_DIR)/*.aux
	rm -rf $(OUT_DIR)/*.log
	rm -rf $(OUT_DIR)/*.toc
	rm -rf $(OUT_DIR)/*.out
	rm -rf $(OUT_DIR)/*.fls
	rm -rf $(OUT_DIR)/*.fdb_latexmk
