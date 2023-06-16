# patternsfromattacks

## Description

This repository contains the necessary python files to use the **Adversarial Classifier**.
It is a decoder that takes advantage of gradient based adversarial attacks in order to 
probe the decision hyperplane. Models built on Pytorch and attacks built on Foolbox are 
used. (**model architecture could be imported in the future by the user**)
The decoder is binary, meaning that only two classes of data can be classified at once. 


## Installation

The main requirement for the installation of the Adversarial Classifier is Anaconda 
(v.4.10.1 or later). After installing the prefered Anaconda version, the user should
download the Anaconda environment provided within this repository (**adversarial.yml**).
All required packages are included within this environment. 
The user can create a new enviromnet using this file using the following terminal command:
	
	conda env create -f adversarial.yml

The user can then activate the environment using the following terminal command:

	conda activate adversarial

The adversarial classifier should always be used within this environment. 
The code is writen using Python v.3.8.8.


## Documentation

Documentation for each function can be found within the code files. 


## Input structure

The user should provide the dataset to be analysed in a similar format to the 
provided example (hello_world). Specifically, a python dictionary is needed with two entries.
The dictionary keys for the two entries should be **data** and **labels** respectivally.
Each of the above entries will hold a list. The **data** list is composed of several 
Numpy arrays,one per trial. Each array should be a matrix of size (N,D) with N the amount of 
patterns and D the dimensionality of the trial. The **labels** list is composed of integers
, one per trial. Since the adversarial classifier is binary, two types of trials can only be
analysed at once and consequently only two unique integer labels should be found within the 
**labels** list. 
Please have a look at the hello_world example dataset for further comprehension of the input
dataset required structure.


## Quickstart

Once the input dataset is appropriatelly structured, the used should dowload all the code files 
located in the **code** directory within the repository.
The user should fill out the desired parameters for the dataset within the **parameters.txt** file.
Further information regarding the parameters can be found within the .txt file.
The **parameters.txt** file should be located in the same directory as the code. 
(**in the future a simple GUI could be introduced for the essential parameters innitialization**)

The user can initiallize the permutation test using the following terminal command:
(assuming the environment mentioned above is already activated)

	python3 significance_play.py

Once the test is finished the results will be stored in the same directory as the dataset.
The user can then initiallize the adversarial classifier using the following terminal command:

	python3 attack_play.py

The results will be stored in the form of a python dictionary within the previously mentioned directory.
Finally the user can initiallize the relevance test using the following terminal command:

    python3 relevance_play.py





 



 
