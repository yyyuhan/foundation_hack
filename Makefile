SHELL=/bin/bash
LINT_PATHS=src/ tests/

lint:
	flake8 ${LINT_PATHS} --count --exit-zero --statistics --max-line-length 127 --ignore=W503,W504,E203,E231

format:
	# Sort imports
	isort --profile black ${LINT_PATHS} 
	# Reformat using black
	black -l 127 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check --profile black ${LINT_PATHS}
	# Reformat using black
	black --check -l 127 ${LINT_PATHS}

commit-checks: check-codestyle lint
