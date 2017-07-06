# Neural Bag-of-Features for EGG classification
Using Bag-of-Features (BoF) to classify EEG time-series data

This repository demonstrates how to use the [Neural BoF model ](https://github.com/passalis/neural-bof) to classify time-series data. In contrast to other well-known models tailored for time-series classification, the BoF model discards most of the spatial information contained in the time-series. This can be especially advantageous when we want to detect certain features in a time-series (e.g., EEG, ECG, etc).

The supplied code evaluates the following models:

| Model         | Accuracy |
| ------------- | ------------- |
| MLP           | 73.2 % |
| GRU           | 76.2 % |
| BoF           | 67.3 % |
| Neural BoF    | 86.7 % |

If you use this code in your work please cite the following paper:

<pre>
@inproceedings{neural-bof-eeg,
        title       = "Time-series Classification Using Neural Bag-of-Features",
	author      = "Passalis, Nikolaos and Tsantekidis, Avraam and Tefas, Anastasios and Kanniainen, Juho and Gabbouj, Moncef and Iosifidis, Alexandros",
	booktitle   = "Proceedings of the 25th European Signal Processing Conference",
	pages       = "TBA",
	year        = "2017"
}
</pre>


Also, check my [website](http://users.auth.gr/passalis) for more projects and stuff!
