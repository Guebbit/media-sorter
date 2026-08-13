# Everything runs in a local virtualenv. `make install` is the only target that
# needs the network; after that the tool talks to nothing but your own Ollama.

PYTHON        ?= python3
VENV          ?= .venv
BIN           := $(VENV)/bin
MEDIASORT     := $(BIN)/mediasort
# cu128 = NVIDIA, cpu = no GPU. See README > GPU.
# Not cu124: that index has no wheels for Python 3.13+.
TORCH_VARIANT ?= cu128
TORCH_INDEX   := https://download.pytorch.org/whl/$(TORCH_VARIANT)
# What `make seed` puts in a rules file that does not exist yet. Override to
# start from your own classes: make seed DEMO_CLASSES=cat,bird,video
DEMO_CLASSES  ?= person,cat,dog,video
# Where `make link` puts the symlink. The usual per-user bin folder.
BINDIR        ?= $(HOME)/.local/bin

.PHONY: help install install-cpu seed link unlink uninstall freeze doctor config scan detect \
        analyze run web rules preview recheck test apply stats verify export clean-links reset

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
	@$(MAKE) --no-print-directory seed
	@echo
	@echo "installed. next:"
	@echo "  $(MEDIASORT) config set --input /path/to/photos   # the one thing it cannot guess"
	@echo "  make doctor                    # GPU, weights, Ollama, rules, paths"
	@echo "  make web                       # rules editor on 127.0.0.1:8765"
	@echo
	@echo "the commands above are wrappers around $(MEDIASORT). To type"
	@echo "\`mediasort\` yourself: \`source $(BIN)/activate\`, or \`make link\` once."

install-cpu:  ## Same, without CUDA (much smaller download)
	$(MAKE) install TORCH_VARIANT=cpu

seed:       ## Write a starter ruleset if there is none yet (safe to re-run)
	@$(MEDIASORT) rules init --classes $(DEMO_CLASSES) \
		|| echo "keeping the rules already on disk — nothing was changed"

link:       ## Put `mediasort` on your PATH: a symlink in ~/.local/bin (BINDIR=)
	@mkdir -p $(BINDIR)
	@ln -sfn $(CURDIR)/$(MEDIASORT) $(BINDIR)/mediasort
	@echo "linked $(BINDIR)/mediasort -> $(CURDIR)/$(MEDIASORT)"
	@case ":$$PATH:" in \
		*":$(BINDIR):"*) echo "run \`mediasort --help\` from anywhere." ;; \
		*) echo "note: $(BINDIR) is not on your PATH. Add it to ~/.zshrc:" ; \
		   echo "  export PATH=\"$(BINDIR):\$$PATH\"" ;; \
	esac
	@echo "the link points into this folder: move or rename it and re-run \`make link\`."

unlink:     ## Remove that symlink again
	@if [ -L $(BINDIR)/mediasort ]; then rm $(BINDIR)/mediasort; \
		echo "removed $(BINDIR)/mediasort"; \
	else echo "nothing to remove: no symlink at $(BINDIR)/mediasort"; fi

uninstall:  ## Delete the virtualenv. Photos, index and rules are untouched.
	rm -rf $(VENV)
	@echo "if you ran \`make link\`, \`make unlink\` removes the dangling symlink."

freeze:  ## Record the exact versions currently installed
	$(BIN)/pip freeze > requirements.lock.txt

doctor:     ## Check GPU, weights, Ollama connectivity and paths
	$(MEDIASORT) doctor

config:     ## Show the photo folders, output folder and Ollama settings in force
	$(MEDIASORT) config show

scan:       ## Index new/changed photos
	$(MEDIASORT) scan

detect:     ## Detection pass only
	$(MEDIASORT) detect

analyze:    ## Semantic pass only (needs your Ollama running)
	$(MEDIASORT) analyze

run:        ## Full pipeline: scan -> detect + analyze -> apply rules
	$(MEDIASORT) run

web:        ## Rules editor / control panel (http://127.0.0.1:8765)
	$(MEDIASORT) web

rules:      ## Show the active rules in evaluation order
	$(MEDIASORT) rules show

preview:    ## Dry run: what would happen, without writing anything
	$(MEDIASORT) apply --dry-run

recheck:    ## Re-apply rules to existing detections (no GPU needed)
	$(MEDIASORT) recheck

apply:      ## Execute the rule actions (links; deletions need --yes)
	$(MEDIASORT) apply

stats:      ## Show progress
	$(MEDIASORT) stats

verify:     ## Check originals and links still exist
	$(MEDIASORT) verify

export:     ## Dump the index to the output folder
	$(MEDIASORT) export

test:       ## Run the test suite
	$(BIN)/pytest -q tests

clean-links: ## Delete the generated output tree (originals untouched)
	$(MEDIASORT) clean --links --yes

reset:      ## Reprocess everything from scratch (keeps the file index)
	$(MEDIASORT) reset all --yes
