IMAGE ?= camex-multicamera-tracking:v0.1.0-poc
DATA_DIR ?=
DEMO_DURATION ?= 60

.PHONY: build models setup test lint demo audit

build:
	docker build -t $(IMAGE) .

models:
	docker run --rm --gpus all \
		-v "$(CURDIR)/models:/app/models" \
		$(IMAGE) python scripts/setup_models.py

setup: build models

test:
	docker run --rm $(IMAGE) python -m pytest -q tests

lint:
	docker run --rm $(IMAGE) ruff check src tests examples scripts

demo:
	@test -n "$(DATA_DIR)" || (echo "Set DATA_DIR to the private three-camera input directory" >&2; exit 1)
	mkdir -p outputs/demo_3cam
	docker run --rm --gpus all \
		-v "$(CURDIR)/models:/app/models:ro" \
		-v "$(DATA_DIR):/demo-data:ro" \
		-v "$(CURDIR)/outputs:/app/outputs" \
		$(IMAGE) python examples/run_three_camera_demo.py \
			--data-dir /demo-data --duration $(DEMO_DURATION)

audit:
	bash scripts/audit_release.sh
