VERSION ?= $(shell uv version --short)
REGISTRY ?= registry.gitlab.com/mikedoesdatascience/sportsml

PLATFORM ?= cpu

default: build

get-version:
	@echo $(VERSION)

build:
	@docker build \
		--build-arg=PLATFORM=$(PLATFORM) \
		-t $(REGISTRY):$(VERSION)-$(PLATFORM) \
		-f docker/Dockerfile \
		.

build-prod:
	@docker build \
		--build-arg=PLATFORM=$(PLATFORM) \
		--build-arg=VERSION=$(VERSION) \
		-t $(REGISTRY):$(VERSION)-$(PLATFORM) \
		-f docker/Dockerfile.prod \
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