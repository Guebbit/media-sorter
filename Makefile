# Everything runs in a local virtualenv. `make install` is the only target that
# needs the network; after that the tool talks to nothing but your own Ollama.

PYTHON        ?= python3
VENV          ?= .venv
BIN           := $(VENV)/bin
PHOTOSORT     := $(BIN)/photosort
# cu128 = NVIDIA, cpu = no GPU. See README > GPU.
# Not cu124: that index has no wheels for Python 3.13+.
TORCH_VARIANT ?= cu128
TORCH_INDEX   := https://download.pytorch.org/whl/$(TORCH_VARIANT)

.PHONY: help install install-cpu uninstall freeze doctor config scan detect analyze run web rules \
        preview recheck test apply stats verify export clean-links reset

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(BIN)/pip:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(BIN)/pip  ## Create .venv and install everything (downloads torch, ~2.5 GB)
	@echo "torch ($(TORCH_VARIANT)) first, so ultralytics cannot pull a different build over it"
	$(BIN)/pip install --index-url $(TORCH_INDEX) torch torchvision
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt
	$(BIN)/pip install --no-deps -e .
	@echo
	@echo "installed. next: $(PHOTOSORT) config set --input /path/to/photos && make doctor"

install-cpu:  ## Same, without CUDA (much smaller download)
	$(MAKE) install TORCH_VARIANT=cpu

uninstall:  ## Delete the virtualenv. Photos, index and rules are untouched.
	rm -rf $(VENV)

freeze:  ## Record the exact versions currently installed
	$(BIN)/pip freeze > requirements.lock.txt

doctor:     ## Check GPU, weights, Ollama connectivity and paths
	$(PHOTOSORT) doctor

config:     ## Show the photo folders, output folder and Ollama settings in force
	$(PHOTOSORT) config show

scan:       ## Index new/changed photos
	$(PHOTOSORT) scan

detect:     ## Detection pass only
	$(PHOTOSORT) detect

analyze:    ## Semantic pass only (needs your Ollama running)
	$(PHOTOSORT) analyze

run:        ## Full pipeline: scan -> detect + analyze -> apply rules
	$(PHOTOSORT) run

web:        ## Rules editor / control panel (http://127.0.0.1:8765)
	$(PHOTOSORT) web

rules:      ## Show the active rules in evaluation order
	$(PHOTOSORT) rules show

preview:    ## Dry run: what would happen, without writing anything
	$(PHOTOSORT) apply --dry-run

recheck:    ## Re-apply rules to existing detections (no GPU needed)
	$(PHOTOSORT) recheck

apply:      ## Execute the rule actions (links; deletions need --yes)
	$(PHOTOSORT) apply

stats:      ## Show progress
	$(PHOTOSORT) stats

verify:     ## Check originals and links still exist
	$(PHOTOSORT) verify

export:     ## Dump the index to the output folder
	$(PHOTOSORT) export

test:       ## Run the test suite
	$(BIN)/pytest -q tests

clean-links: ## Delete the generated output tree (originals untouched)
	$(PHOTOSORT) clean --links --yes

reset:      ## Reprocess everything from scratch (keeps the file index)
	$(PHOTOSORT) reset all --yes
