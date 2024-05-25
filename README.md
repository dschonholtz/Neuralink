# A fun challenge.

## A preface

This is the first time I've tried to write C or C++ in about 7 years... It is a bit rough.

## What is next

We have to rewrite huffman so that we can calculate a general huffman table for all of the values in the dataset.
Anything that is in the int16 range that is not seen will be given a frequency of 1 in the table.

We will base this off of the first 20% of the dataset and see how it performs on the remaining 80%.

I also haven't set up GDB or other debugging tools and it really sucks.
Might switch to clion for that reason.

Then I just bundle in the huffman table with the encode binaries.

Then I just run it in linear time should be blazing fast and power efficient.

I still don't get how I see a 2.5 compression ratio in python and a 30% improvement in c++ more me not knowing how to write C++ I guess.
That is only the first binary. It is likely many of the other binaries have better compression ratios. Assume that the implementation is roughly correct and get it to work losslessly.

So concrete steps are.

1. Save huffman tree to disk.
2. Load from disk
3. Use clion to debug the tests and then find out why the stupid huffman encoding is resulting in larger files than it should.
4. Submit this whole repo to neuralink.
