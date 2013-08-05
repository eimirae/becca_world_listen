becca_world_listen
==================

The **listen** world identifies anomalies or unexpected occurences in audio data. It does this by building the data into a hierarchy of features and learning patterns of feature activity. 

To run BECCA with the listen world, clone this repository into your local copy of the `becca` repository so that it sits in the same directory as the `core` and `worlds` directories.

In `tester.py`, add the line:
```
from becca_world_listen.listen import World
```
and comment out all other World import lines.

Typing `python tester.py` at the command line will run the listen world. It draws training data from all `.txt` files that it finds in the `becca_world_listen/data` directory. 

In `listen.py`, manually set the flag variable `self.TEST` to `True` when you want to test the anomaly detection performance. It will look for the test data in `becca_world_listen/test/test.txt` and for ground truth information in the same directory under `truth.txt`. 

The audio text file format for both training and test data is a single ASCII number in double format (e.g. '3.4847658437e-004') per line of the file. Each line of the file represents one audio sample. 

The ground truth text file format is two ASCII numbers per line (e.g. 12.3 14.7) indicating the start and stop times of an anomaly in seconds. There is one line per anomaly.
