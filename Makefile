REGISTRY ?= registry.gitlab.com/mikedoesdatascience/sportsml
VERSION ?= 0.0.1

default: build

freeze:
	@docker build \
		-t $(REGISTRY):freeze \
		-f docker/Dockerfile.freeze \
		.
	@docker run -it --rm \
		-v $(shell pwd):$(shell pwd) \
		-w $(shell pwd) \
		-u $(shell id -u):$(shell id -g) \
		$(REGISTRY):freeze \
			pip-compile python/pyproject.toml --resolver=backtracking --no-annotate --no-header --cache-dir /tmp/.cache -o python/requirements.txt
	

build:
	@docker build \
		-t $(REGISTRY):$(VERSION) \
		-f docker/Dockerfile \
		.

push:
	@docker push $(REGISTRY):$(VERSION)

debug:
	@docker run -it --rm \
		-v $(shell pwd)/python/sportsml:/usr/local/lib/python3.10/site-packages/sportsml \
		-e MONGODB_URI \
		-e MONGODB_USERNAME \
		-e MONGODB_PASSWORD \
		$(VOLUMES) \
		-w /project \
		--entrypoint bash \
		$(REGISTRY):$(VERSION)