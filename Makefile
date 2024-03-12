ifneq (,$(wildcard ./.env))
    include .env
    export
endif

cmd-exists-%:
	@hash $(*) > /dev/null 2>&1 || \
		(echo "ERROR: '$(*)' must be installed and available on your PATH."; exit 1)

init: cmd-exists-poetry cmd-exists-python cmd-exists-pip
	pip install poetry
	poetry config virtualenvs.in-project true
	poetry install
	cd notebooks && poetry install --no-root