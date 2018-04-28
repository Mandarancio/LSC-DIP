# Configuration Folder

The configuration used in the experiments are stored here in format of  `YAML` file.

Example of configuration:

```yaml
# data folder
datafolder: "../datasets/"
# dataset name
dataset: "YaleB"
# image format
fformat: ".png"
# resize option
resize: none
# number of bands
nbands: 4
# options: default (np.reshape(-1)), zigzag, spiral and circular
vectorization: zigzag
# training option
training:
    # options: none, energy and random
    sort: none
    # number of levels for discretization of the spectrum (see Sec. A.2)
    n_levels: 100
    # number of codes per band
    n_codes: 250
    # batch size
    batch: 100
    # show stats: NOT USED
    show_stats: true
# testing option
# to disable one testing step simply remove it from the configuration
testing:
    # sampling rate vs reconstruction test
    sampling_plot:
        # sampling log-range: start, stop, n. steps
        sampling_range:
            - -2
            - 0
            - 20 
    # visual results test
    visual_results:
        # options: random (1 random sample), array(....) list of sample
        samples: random
        # sampling log-range: start, stop, n. steps
        sampling_range: 
            - -3
            - 0
            - 20        
    # robust reconstruction at fixed sampling rate test
    robust_reconstruction:
        # sampling rate
        sampling_rate: 0.05
        # noise variance log-range: min, max, n. stop
        noise_range: 
            - -2
            - 2
            - 20
    # sampling at fixed noise level
    robust_sampling:
        # noise variance
        noise_rate: 15
        # sampling log-range: start, stop, n. steps
        sampling_range:
            - -2
            - 0
            - 20 
        # sampling at fixed noise level
    # robust reconstruction at fixed sampling rate visual test
    robust_sampling_visual:
        # sampling rate
        sampling_rate: 0.05
        # noise variance log-range: min, max, n. stop
        noise_range: 
            - -2
            - 2
            - 20
        # options: random (1 random sample), array(....) list of sample
        samples: random
```