# Simple CMake wrapper Makefile

BUILD_DIR := build
CMAKE     := cmake

.PHONY: all configure build run test clean rebuild

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR)

build: configure
	$(CMAKE) --build $(BUILD_DIR)

run: build
	./$(BUILD_DIR)/nn_demo

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean || true

rebuild:
	rm -rf $(BUILD_DIR)
	$(MAKE) build
