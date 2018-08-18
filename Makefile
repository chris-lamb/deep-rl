.PHONY: build
build:
	docker build -t deep-rl:current .

.PHONY: run
run:
	docker run -it --rm -v `pwd`:/deep-rl deep-rl:current
