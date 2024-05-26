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

Huffman is officially a dead end....
Deflate, used in zip uses huffman and a sliding window technique under the hood.

The reason I was able to get so far and get a 2.53 number (in python still no luck in c++)
is because I am building an optimal huffman tree for that specific wav file.
To actually use it for this usecase you would have to then ship the huffman tree with the data or use a fixed huffman tree.

Looking at the data distribution, it is too wide and peaky and huffman isn't efficient enough to actually beat the naive zip implementation which undoubtedly does one of those things with DEFLATE.

## Ok what if we use neural networks even though we have to be lossless?

Hear me out, I have spent the last two years working with iEEG data off and on to do seizure prediction.
We specifically do not attempt to transmit the raw data and do ML elsewhere because the transmission costs
neuralink is running into are prohibitively expensive. They of course MUST ship the data because they want to do arbitrary analysis.

To get around this for seizure prediction, we just run a neural network on the chip in the persons head, or transmit the data to another device in the persons body via sonar and then do the ML there via very low power.
We run a very small model that does all of the processing that way. This allows us to do deep brain stimulation only when we need to to prevent seizures.

So what if we do the same thing here, except instead of a specific seizure prediction algorithm we run a neural network to do compression.

Now of course the correct reaction to that is to say neural networks only are fitting a function to the data, the second you have data the model misclassifies or have data drift then this will not work.

But we can get around that.

We can run the model locally on the chip, do validation the model is predicting the correct thing, then output either the very heavily compressed output from the network, or if the network is wrong we can stream the full representation of the number.

Here is what I am thinking.

For our channel, we take the previous 5-10 samples which would look like the below raw data (notice how regular the binary looks)

0000001010100000
0000001111100000
0000011011100001
0000001111100000
0000000101011111
0000001001100000
0000001001100000
0000000100011111
0000000111011111
0000001100100000
0000010000100000
0000001010100000
0000000111011111
0000000110011111
0000000001011111
0000000000011111

Then for any given value and it's previous values we have a look up table of the 8,16,32,64 whatever most likely outcomes for that previous set of values.

We then do the look up of the position the next value is in the likelihood table. If it isn't there we transmit the full value. If it is we just have to transmit the position it is in in the likelihood table as an unsigned int.
