.PHONY: install test smoke

install:
	python -m pip install -e .

test:
	python -m pytest

smoke:
	python examples/j2_precession.py --orbits 40
