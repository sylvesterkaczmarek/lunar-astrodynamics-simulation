.PHONY: install test smoke download-grgm1200a

install:
	python -m pip install -e .[dev]

test:
	python -m pytest

smoke:
	python examples/j2_precession.py --orbits 40
	python examples/harmonic_validation.py

download-grgm1200a:
	python scripts/download_grgm1200a.py
