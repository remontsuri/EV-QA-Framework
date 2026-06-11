.PHONY: docs docs-clean test lint

# Build Sphinx documentation
docs:
	uv run sphinx-build -b html docs/source docs/build/html

# Clean and rebuild docs
docs-clean:
	rm -rf docs/build
	uv run sphinx-build -b html docs/source docs/build/html

# Check docs for broken links
docs-linkcheck:
	uv run sphinx-build -b linkcheck docs/source docs/build/linkcheck
