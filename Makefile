ifneq (,$(wildcard ./.env))
    include .env
    export
endif

cmd-exists-%:
	@hash $(*) > /dev/null 2>&1 || \
		(echo "ERROR: '$(*)' must be installed and available on your PATH."; exit 1)

init: cmd-exists-python cmd-exists-pip
	python3 -m venv .venv
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install poetry && \
	poetry config virtualenvs.in-project true && \
	pip install poetry-plugin-export && \
	poetry install && \