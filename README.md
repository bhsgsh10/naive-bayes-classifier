# naive-bayes-classifier
 Implementation of a Naive Bayes Classifier for classifying words as spam or non-spam.   
 Use the following command for running the project:   
 ```$ python classifier.py train.txt test.txt```
 
```collect_attribute_types()```, takes in the name of a training dataset file and an integer m. It finds a vocabulary consisting of the set of unique words occurring at least m times in the training data and stores it as a set in ```self.attribute_types```.   
```train()``` is used to train the classifier.   
```predict()``` takes the text of an input SMS and returns a dictionary of relative beliefs for the classification of the input to each label(spam/not spam).   
```evaluate()``` takes the filename for a dataset, classifies each individual example in the file using predict(), and returns the overall classification accuracy. Accuracy figures are mentioned in comments in the Python file.   
The classifier can be tuned using different values of `k(smoothing parameter)` and `m(cutoff)`.   
Stopwords, which frequently occur in the English text should not be counted among words which occur the most in the given text. ```remove_stopwords()``` removes such words from ```self.attribute_types```.
