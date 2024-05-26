# Neuralink Compression: 2.54 with Optimal Huffman Coding, and 2.268 or .6% better than zlib with one hop look up

This work and readme is based off of my last 48 hours of learning C++ and compression for the first time in 8 years, exploring the data, and my last two years off and on doing seizure prediction with iEEG data at the Northeastern Wireless Networks and Embedded Systems Lab resulting in a couple papers.

Please note, that all of this was done in under 50 hours.
So bear with me, it is rough around the edges.

In this readme, you'll learn the following.

1. How I got a score of 2.54 with huffman coding,
2. Why you won't be able to use that neuralink
3. How I built a slightly smaller in scope algorithm and combined it with zlib to beat the default zip performance which hopefully will top the leaderboards once I go and implement it in c++.
4. How I explored the data both in jupyter notebooks and when staring at the raw binary and how it will be able to help you at neuralink.
5. I will explain with a proof why lossless 200x compression is impossible with this dataset
6. I will explain how you can get to significantly higher compression ratios if you provide further data and/or allow som level of lossiness.
7. Finally, I will explain how I would attempt to solve this problem if I was working at neuralink and had full access to your data and ML infra based on my last two years of working with similar data and ML pipelines.
8. Why you should please, please please hire me. Although maybe for ML on embedded systems rather than as a C++ dev.

## Understanding the data and compression

First piece of critical thinking. Is that most likely, neither I nor anyone else in this challenge will invent an entirely new general compression algorithm. Any performance gains we make will be from finding abnormal structure in the data, then building a compression algorithm that abuses that structure in a way that allows us to get significantly higher compression ratios than you would from a completely general algorithm.

You can see how I initially explored the data in [playground.ipynb](https://github.com/dschonholtz/neuralink/blob/main/playground.ipynb)

Here are some basic stats:
743 monochannel WAV files all sampled at 19531 Hz

The data type is int16 with values ranging from -32768, 32767 (the max vals for int16s)

The smallest non-zero step size between two consecutive values is 63.

This is our first big hint, that screams a power of 2 and that maybe we can shave off 6 bits on every single value. 2^ 6 = 64

If we plot the histogram of all possible values we learn a lot!
The values are in an approximate normal distribution, and the cast majority of them are clustered. This is good! More structure = higher compression.

![Histogram of All Data](Images/HistogramOfAllData.png)

As we look closer though it is a bit concerning that the data is not as normal as we would ideally like, and the distribution is wider than we would like. Ideally, we'd have 90+% of the signals in just 5% of the values.

Instead we see:

75% of the data falls between -801.0 and 2658.0
85% of the data falls between -1377.0 and 3234.0
95% of the data falls between -2851.0 and 4643.0
99% of the data falls between -10026.0 and 12971.0

Part of the importance here comes from thinking about bitwise representation of numbers. If we can represent the vast majority of the numbers with fewer bits, then we should be able to generate higher compression ratios.

The difference between representing all of the data and 75% of the data is 4 bits per value, this certainly isn't nothing, but it isn't as much as one would hope if you are trying to beat 2.2x compression.

2048, 4096, 8192, 16384, 32768, 65536

Nevertheless, this is where I started.

I compared huffman coding(from scratch), zip, mp3 (I know it is lossy, I was curious), FLAC, and a look up compressor (You don't recognize this name because I made it up).

I did all of this initially in python so it isn't compatible with the neuralink test suite, but I wrote my own.

I initially got these values:

| Method  | Compression Ratio | Notes                                    |
| ------- | ----------------- | ---------------------------------------- |
| Huffman | 2.54              | 15% better than zip!                     |
| Zip     | 2.2               |
| Mp3     | 9.535             | Compression is ez if you can delete data |
| FLAC    | 1.489             |
| Look Up | 2.268             | (More on this later)                     |

Let's back up a second, and explain why I chose Huffman, what it does, and why I thought it might do well here. Then we'll explain why this might be too good to be true.

Huffman, encodes common values via a tree with smaller binary representations.
This means that your most common values take the least bits.

And we saw in our distribution of data, we have a lot of common values!

In my implementation in python, I built an optimal huffman tree for each 5 second data source individually. This meant that every piece of data had a representation in the huffman tree, and that nothing else did.

The obvious problem there, that I didn't realize until after implementing it, is that this means that if you must decode in a separate place from where you encode, you either have to have a common pre-determined tree, or you must transmit the entire tree!

So I attempted to build a huffman implementation that is based off of the entire distribution of all of the data, and found that this of course is outperformed by zip by decent margin. When we peak under the hood at the zip algorithm, DEFLATE, this makes sense too. DEFLATE uses huffman trees and combines it with a form of run length encoding to get such a good general performance.

I was almost ready to give up, but then I was briefly looking at the raw binary and saw the sinusoidal pattern you normally see in EEG data.

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

This rough pattern goes on for varying lengths.

But my thought is that any individual value seems to only either go up, down or repeat the same value.

The thought is there is a chance, that we can store a look up table for each individual value that corresponds to a small subset of values. That is much more efficient than you would get from a general algorithm.
As if you can represent 32 possible next values, then you only need
5 bits to do that.

So I built something that naively looks at the current value, and then looks at the most common next possible value for the entire dataset.

I add a bit in front of this bit string to signify if it exists or not in the look up table. Then I tested look up tables for each value of length 32, 64, 128 and 256.

128 consistently won with a compression ratio of 1.91, something ok, but not incredible.

However, this data compression is likely to still be highly compressible by other methods.

So then we combine that with zlib again.

That gets us to a whopping 2.268. Or about 0.6% improvement over zlib. This is a technical improvement, but not by much.

We can potentially adjust the look up table to be for pairs, but that starts to look like the combination of run length encoding paired with huffman coding which of course is what zip does.

## What is next?

So after experimenting... Let's take a step back and look at what is even possible given this dataset.

Let's look at these requirements again and see if we can identify anything else that might be making this harder than it needs to be.

1 ms latency and 20khz with 200x compression.

Well in a ms we only get 20 samples of 10 bits each.
This means that the at 200x compression each sample must be compressed to a single bit.

That is trivially impossible as any data that is out of distribution no matter how much you know about the previous data obviously cannot be compressed to a single bit.

So with this dataset that is impossible, but how far could we get with a better dataset? Could we do better? For instance, if we get 1000 channels at once, could we compress across channels?

I looked at the EDF EEG dataset here: https://physionet.org/content/chbmit/1.0.0/

I also have access to iEEG data, but cannot share it due to IP concerns, but if you are interested, you can read or get the data yourself from epilepsae: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3296971/. I am also working on a data pipeline for EDF EEG scalp data and that data is open source and from MIT so we can look at that.

![EDF Heatmap](Images/HeatmapEdf.png)

We can see that different channel have between 1 and 5% overlap with other channels generally. 6 of the 23 channels had overlap of > 3%. This is significantly higher than you would expect for the int16 values, but when looking at the distribution of data, we can see that it is just a much smaller range and this largely entirely explains the overlapping values.

![EDF Histogram](Images/EdfHistogram.png)

The naive interpretation of this suggests at least for the EDG scalp EEG dataset, the values are not strongly correlated, and to know if we could further compress data with correlation between neighboring signals, we would have to look directly at the full neuralink data, but it doesn't not look particularly promising given these preliminary results.

# This problem doesn't seem tractable. How do we solve it anyway?

So far, we have learned a lot, but haven't made much headway. How could we make progress? Well, the obvious answer would be to use a neural network to do the compression and to accept some loss. The neuralink team probably doesn't want to do this, because they do not want to build a compression algorithm that drops data that their ML team might pick up on, but the ML team is likely compressing this data in their model anyway for specific tasks.

On top of this, in academia, this exact problem, transmission being too expensive is the exact reason the lab I have worked in does the ML in chip in your head rather than attempting to transmit the data to an external source. The problem is, since neuralink want to eventually do arbitrary data processing with this data, you need to be able to transmit it all somewhere else. This has other problems like the data often needs to be trained per person, it is hard to do transfer learning well between patients, but it still at least appears to be possible.

The solution most likely looks like an auto encoder network as pictured below.

![Auto Encoder](Images/AutoEncoder.png)

You train a network on a massive amount of raw data that has a very highly compressed latent representation of that data and then the network expands that back out into an image or in this case the matrix of 1024 sensor outputs.

What is tricky about this, is that this is closer to video, so the auto encoder architecture likely needs to be closer to a vision transformer architecture or a series of 1d CNNs/LSTMs or transfomers that are then fed into a fairly normal auto encoder. To learn how to actually do this well would require reading a lot of papers on video auto encoders and do further analysis on the architecture, size and if the models can be quantized down onto a chip like the one in the neuralink.

The problem I expect you would run into, is even after aggressively pruning and quantizing the model, you still can't realistically fit it on your chip.

So what you might end up doing is for a single neuron at a time or for a group of them just attempt to predict the next value in the time series data.
This model could be very small, and be run on a chip like the neuralink, although an embedded GPU would help, and should be accurate judging by our seizure prediction results: https://arxiv.org/abs/2401.06644

Then you could check if the predicted result is correct.
Then you would only have to transmit the values and the locations in the array where the value is not correct, then on the receiver, you would run the exact same model to decompress everything except for the values you transmitted.

We experimented with a bunch of loss functions and model architectures and the 1D CNN with a focal loss function performed the best for seizure prediction for a single channel and I'm guessing it may perform well in predicting the next value for a stream as well.

The problems here come back to being patient specific, and having to train a new model for each patient, but maybe this could be overcome with transfer learning, hence my not giving you an implementation right now.

If you gave me data and some GPUs then I'd love to give it a whirl though. I'll also run this by my lab and see if I can do this in parallel with the data pipeline stuff I'm currently working on.
