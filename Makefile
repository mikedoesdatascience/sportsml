REGISTRY ?= registry.gitlab.com/mikedoesdatascience/sportsml
VERSION ?= $(shell cd python && python3 setup.py --version)

default: build

pip-lock:
	@docker build \
		--no-cache \
		-t $(REGISTRY):lock \
		-f docker/Dockerfile.lock \
		.
	@docker run -it --rm \
		-v $(shell pwd):$(shell pwd) \
		-w $(shell pwd) \
		-u $(shell id -u):$(shell id -g) \
		$(REGISTRY):lock \
			python -m pip freeze --no-cache-dir > python/requirements.lock
	

build:
	@docker build \
		-t $(REGISTRY):$(VERSION) \
		-f docker/Dockerfile \
		.

build-prod:
	@docker build \
		-t $(REGISTRY):$(VERSION) \
		-f docker/Dockerfile.prod \
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
		--shm-size=8gb \
		$(REGISTRY):$(VERSION)

run:
	@docker run -it --rm \
		-e MONGODB_URI \
		-e MONGODB_USERNAME \
		-e MONGODB_PASSWORD \
		$(VOLUMES) \
		-w /project \
		--entrypoint bash \
		$(REGISTRY):$(VERSION)

upload:
	@docker run -it --rm \
		-e MONGODB_URI \
		-e MONGODB_USERNAME \
		-e MONGODB_PASSWORD \
		$(REGISTRY):$(VERSION) \
			nba_mongo_upload

	@docker run -it --rm \
		-e MONGODB_URI \
		-e MONGODB_USERNAME \
		-e MONGODB_PASSWORD \
		$(REGISTRY):$(VERSION) \
			nfl_mongo_upload