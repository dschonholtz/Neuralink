.PHONY: all build clean build_huffman_tree

all: build

build:
	cd cpp && mkdir -p build && cd build && cmake .. && make

clean:
	rm -rf cpp/build

build_huffman_tree: build
	./cpp/build/diskHuffmanTree data huffman_tree.dat


test:
	./cpp/build/testHuffman
