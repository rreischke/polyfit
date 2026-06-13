PYTHON  ?= python3
APP_DIR ?= ~/Applications

.PHONY: install uninstall reinstall

## Install: pip install + create platform-specific launcher
install:
	$(PYTHON) create_app.py "$(APP_DIR)"

## Uninstall: remove launcher files + pip uninstall
uninstall:
	$(PYTHON) create_app.py --uninstall "$(APP_DIR)"

## Reinstall from scratch
reinstall: uninstall install
