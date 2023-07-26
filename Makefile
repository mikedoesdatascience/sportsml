REGISTRY ?= registry.gitlab.com/mikedoesdatascience/sportsml
VERSION ?= $(shell cd python && python3 setup.py --version)

PLATFORM ?= cpu

default: build

pip-lock:
	@docker build \
		--no-cache \
		-t $(REGISTRY):lock-$(PLATFORM) \
		-f docker/$(PLATFORM)/Dockerfile.lock \
		.
	@docker run -it --rm \
		$(REGISTRY):lock-$(PLATFORM) \
			> python/requirements.$(PLATFORM).lock
	

build:
	@docker build \
		-t $(REGISTRY):$(VERSION)-$(PLATFORM) \
		--build-arg=PLATFORM=$(PLATFORM) \
		-f docker/$(PLATFORM)/Dockerfile \
		.

build-prod:
	@docker build \
		-t $(REGISTRY):$(VERSION)-$(PLATFORM) \
		--build-arg=PLATFORM=$(PLATFORM) \
		-f docker/$(PLATFORM)/Dockerfile.prod \
		.

push:
	@docker push $(REGISTRY):$(VERSION)-$(PLATFORM)

pull:
	@docker pull $(REGISTRY):$(VERSION)-$(PLATFORM)

debug:
	@docker run -it --rm \
		-v $(shell pwd)/python/sportsml:/usr/local/lib/python3.11/site-packages/sportsml \
		-e MONGODB_URI \
		-e MONGODB_USERNAME \
		-e MONGODB_PASSWORD \
		$(VOLUMES) \
		-w /project \
		--entrypoint bash \
		--shm-size=8gb \
		$(REGISTRY):$(VERSION)-$(PLATFORM)

run:
	@docker run -it --rm \
		-e MONGODB_URI \
		-e MONGODB_USERNAME \
		-e MONGODB_PASSWORD \
		$(VOLUMES) \
		-w /project \
		--entrypoint bash \
		$(REGISTRY):$(VERSION)-$(PLATFORM)

upload:
	@docker run -it --rm \
		-e MONGODB_URI \
		-e MONGODB_USERNAME \
		-e MONGODB_PASSWORD \
		$(REGISTRY):$(VERSION)-$(PLATFORM) \
			nba_mongo_upload

	@docker run -it --rm \
		-e MONGODB_URI \
		-e MONGODB_USERNAME \
		-e MONGODB_PASSWORD \
		$(REGISTRY):$(VERSION)-$(PLATFORM) \
			nfl_mongo_upload