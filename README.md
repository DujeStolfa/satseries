# Dense recognition in satellite image time series

Source code for experiments presented in my Master's thesis titled _Segmentation of satellite image time series by vegetation type_. 

## Abstract

Satellite imagery enables us to monitor and detect undesirable changes in the environment, e.g. the spread of invasive plant species, at very large scales. Given a sequence of spatially-aligned satellite images, taking the temporal context into account can be beneficial for plant species recognition. We consider models that determine sequence-level representations by temporally pooling input sequences. Various attention mechanisms are compared and used to assign a weight to each timestep in the sequence. We evaluate these models on a dataset with labels for the false indigo-bush. Attention pooling outperforms a vanilla transformer encoder despite having substantially fewer parameters. Of all considered approaches, attention with learned queries and values distributed across heads generalizes best to spatial shifts in the data. Despite the coarse spatial resolution of the imagery, taking spatial context into account was beneficial for recognition.

