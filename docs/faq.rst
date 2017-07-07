.. _faq:

Frequently Asked Questions
==========================

Q. What exactly is Rambutan predicting?

A. Rambutan is predicting the probability that two regions in the genome engage in a statistically significant contact, given their nucleotide sequences and DNaseI sensitivity data. Statistical significance is denoted as a q-value (FDR) of 0.01 or less according to Fit-Hi-C.


Q. Most contact matrices are made up of integer counts, why aren't you predicting this?

A. It is true that the output data from a Hi-C matrix comes in the form of an integer number of counts that a pair of regions engage, as determined by deep sequencing runs. However, this number is highly dependent on the sequencing depth of the experiment, which can change from one experiment to the next. In addition, these values suffer from several biases, including a region-specific bias that is influenced by, among other things, GC content, and the very powerful genomic distance effect. Ultimately when considering 3D genome structure one does not think about how many times a pair of regions are in contact according to a sequencing experiment, but simply the binary of whether they are in contact or not. Rambutan's predictions accounts for the biases, and predicts this, arguably more useful measurement of genomic structure.


Q. Can I train Rambutan using a different epigenetic mark?

A. Absolutely. The toolset is already there, since you can feed in an arbitrary bit-encoded epigenetic data in the place of the DNaseI values. The only restriction is that the input must have 8 bits at each position. If you need fewer than 8 bits to encode your epigenetic data it is perfectly acceptable to leave of the bits always off.


Q. Can I use bedgraph_to_dense and encode_dnase to bit encode my different epigenetic mark?

A. Yes, as long as the range of the values is between -2 and 5, or you are fine with limiting the dynamic range of the signal to that width.  If you have some other file that falls within this range that you'd like to use instead, you can treat that signal identically to how you would treat DNaseI.


Q. Why did you restrict the input to only DNaseI signal, won't additional epigenetic marks improve accuracy?

A. It is likely the case that more epigenetic marks will improve accuracy. However, the point of the model is not solely to produce accurate models but to serve as a replacement for running high resolution Hi-C experiments. If a researcher has to run several epigenetic experiments for a cell type that doesn't currently have all of the marks before being able to run Rambutan, it may be worth it for them to simply run the Hi-C experiment. DNaseI signal has been collected for over 400 human cell types, and is cheap and easy to gather for future ones.


Q. How powerful a machine is necessary?

A. The problem is a big data problem so it is unlikely that it can be solved in a reasonable amount of time on a laptop. A user should have at least one GPU and at least 16G of RAM.


Q. Are GPUs necessary to make predictions?

A. Technically it is possible to use only CPUs to make predictions but it is unlikely that this will 


Q. Why is the project named Rambutan?

A. This question is left as an exercise for the user.
