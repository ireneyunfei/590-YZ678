There are a lot of files included in this submission, hope this readme file can help to navigate!

1. Scripts:

As required, there are 3 scripts, 01-clean.py, 02-train.py, and 03-evaluate.py

2. Model Results:

2a. models: CNN.h5, RNN.h5

2b. model log files, which save the training info and final metrics: cnn_log.txt, rnn_log.txt

2c. loss and acc plots: cnn_acc.png, cnn_loss.png, rnn_acc.png, rnn_loss.png

3. Data Files:

Data are downloaded from https://www.gutenberg.org

I only include the raw data and cleaned data fed to the model, otherwise the folder size would be too large. 

I also applied pre-trained model GloVe.6B (100) dimension, which can be found here: https://github.com/stanfordnlp/GloVe. I did not include this in the repo either, again it would make the folder too large.

3a. raw data (texts of 3 books): biology.txt, monte_cristo.txt, and psychology.txt

3b. cleaned data after preprocessing: texts.txt, and labels.csv


Thank you!
