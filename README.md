# Adaboost
An adaboost classifier for identifying the correct orientation of an image. Computer Vision problem

To run the problem, simply type:
python3 orient.py **\<mode> \<test OR train-file.txt> \<model-file.txt> \<model>**

where,
* mode = train or test (you have to train before testing for custom data sets)
* test/train file - a .txt file that contains vectors(192 dimensions) for the images we're training on.
* model-file - Using pickle we store the trained model in this file (*used when classifying testing data*)
* model - enter '**adaboost**' here

#### Note
This program stores the trained model in a temporary file (using pickle) called *model-file.txt*
