# GroundMotionRecordClassifier
Bellagamba et al. (2019) A neural network for automated quality screening of ground motion records from small magnitude earthquakes

Data is provided in the 'data' folder in the form of a csv file and a SQLite database containing the quality metrics and IM used in paper. Note that both files provide the same data.

Note also that a full automation process system is provided in the zip file, but is only valid for GeoNet (NZ) V1A format and has more dependencies that the models alone provided in this git repo.

Dependencies of the classifiers: python 3.X, numpy, os, csv

Perform classification of quality ground motion records as described in Bellagamba et al. (2019) A neural network for automated quality screening of ground motion records from small magnitude earthquakes. Results are stored in a result folder in the same way the data is given by appending the scores yhat_low and yhat_high for resemblance to low and high quality ground motion records, respectively. 

The two models developed in the paper are provided. The authors recommend the use of the Canterbury-Wellington model with a low acceptance threshold (0.5 - 0.6). 

Data must be given as provided in the example, a csv file respecting the following order (numbers indicate the column index after the column to be skipped):

1.  Low frequency (below 0.1Hz) pre-event FAS to maximum signal FAS ratio
2.  Low frequency (below 0.1Hz) entire signal FAS to maximum signal FAS ratio
3.  Minimum SNR
4.  Maximum SNR
5.  Average SNR
6.  Average tail ratio
7.  Maximum tail ratio
8.  Maximum tail noise ratio
9.  Average tail noise ratio
10. Average head ratio
11. Average SNR between 0.1-0.5Hz
12. Average SNR between 0.5-1.0Hz
13. Average SNR between 1.0-2.0Hz
14. Average SNR between 2.0-5.0Hz
15. Average SNR between 5.0-10.0Hz
16. Fourier amplitude ratio
17. Peak noise to PGA ratio
18. 10%–20% bracketed duration ratio
19. 5-75% significant duration
20. 5-95% significant duration

Computation method of each of these metrics are given in the electronic supplement of the paper. The authors recommend the use of obspy to carry out these calculations. 
