.PHONY: build-comparables help

help:
	@echo "Available commands:"
	@echo "  make build-comparables  - Convert rent_ads_rightmove_extended.csv to comparables.csv"

build-comparables:
	@echo "Building comparables dataset..."
	python backend/scripts/build_comparables.py

