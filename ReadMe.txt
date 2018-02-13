Read Me:
%%In order to run it is important to have Matlab2017b and for the python code the necessary libraries

Before running any of the files, the images needs to be preprocessed. For that please run 'prepro.m' file and save the generated results in separate folder. For instance, separated the training set, testing set and the validation set into three folders. Inside training and validation folder, the data of each class must be separated into separated folders

Next,

For training from scratch, change the path of the input data and run the python codes, moeskops_net.py, parallel_net.py. 

For fine tuning, open, fine_tuning_all.m file, change the path of the input files path and run. The corresponding network can be chosen.

For feature extraction, open feature_extraction_all.m file and change the input files paths before running. The corresponding network can be chosen. The extracted features will be saved as a .mat file.

