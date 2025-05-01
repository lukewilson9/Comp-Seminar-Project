Luke Wilson
Kenyon College
Computing Seminar Spring 2025 Final Project

Title: Hierarchical Random Forest Classifier for LIGO Noise

Purpose: Given LIGO data from real gravitational-wave observing runs, the classifier can provide accurate predictions for whether or not a given signal in the data is a gravitational wave or an environmental or outstanding noise artifact (referred to in the field as ‘glitches’). 

General workflow: The random forest classifier is written as a pipeline in a jupyter notebook. ‘Pipeline’ refers to the ability for a scientist or engineer to use this notebook, run it cell by cell down the notebook, and ultimately visualize the model’s classification of their LIGO data signals as either glitches or actual gravitational waves. 

The secondary element to this classifier is its hyperparameter tuning code. This is a hierarchical model, which means we break the data down by subsystems (12 of them), create random forests classifiers for these subsystems and hyperparameter tune them, then use the predictions from these 12 subsystems classifiers as inputs to a overall combiner random forest, which receives hyperparameter tuning, and ultimately provides a final prediction on the classification of a LIGO signal. 

So, this means we have two main units to be used: the pipeline jupyter notebook code, and the hyperparameter tuning code. Both of these notebooks are ideally run with GPU support, but the hyperparameter tuning requires it, considering it is such an intensive computational process.

For a given LIGO dataset, using the model for the first time:

Pull the github and open the .ipynb file in a jupyter notebook on your computer. I included 2 .yml files, which are saved Conda environments that I work out of for the pipeline usage and the hyperparameter tuning. A Conda environment is a self-contained workspace where you can install and manage specific packages and dependencies without affecting your system Python or other projects. Although not necessary to use the pipeline, downloading conda and creating a conda environment on your computer with my .yml files will make it easiest to immediately install all the packages I used to run my pipeline and use my project. For running the pipeline, there is a pipeline.yml file. This is what you should use as your kernel in the jupyter notebook, the default is Python (.ipynb). The following is some code for a conda user to install a conda environment, ‘pipeline’, as a kernel in their jupyter notebook:

# Activate your Conda environment
conda activate pipeline
# Install ipykernel (if not already installed)
pip install ipykernel
# Add the environment to Jupyter with a custom name
python -m ipykernel install --user --name=pipeline --display-name "Python (pipelineEnv)"

Once again, this is not absolutely necessary, but without this, you will have to download all of the libraries and packages I used within the notebook directly.

The first several cells define functions in the pipeline. The data is then balanced and broken out by subsystems. All official LIGO datasets that include proper auxiliary channel information can be broken down this way. The SNR feature is then extracted from the data, any feature could be used technically due to recent developments in feature significance found by the Wade lab, but for this project we just use SNR. At this point, cell 11 is executed.

The next 2 cells help break out and literally export data .csv files of subsystem data. This is necessary for model hyperparameter tuning. After breaking out .csv files of data for all the subsystems, you can use the RSCV_HPT.py script to hyperparameter tune the code. This should be done using an associated GPU. There is a conda environment saved as environment.yml to assist in getting the necessary packages for running this code. Once in this environment and on a GPU, one could run: 

Python RSCV_HPT.py –data (data file).csv &> (text output file).txt

This will actively output the hyperparameter tuning runs for the given subsystem classifier based on that subsystems data. You can dynamically view the file using:

Tail -f (text output file).txt

This should be done for all 12 subsystems in sequence, and the hyperparameter can be recovered from the text files. Parameters in the script can be modified to suit the model being tuned if desired.

Now, the next cell can be executed to get predictions for all 12 subsystem classifiers. There are dictionaries in this cell where you can include the hyperparameters from the tuning. The model will consider those parameters for getting the predictions from each of the 12 classifiers. Then, the code creates a dataset to be used for the combiner predictions.

The next cell exports the subsystem predictions dataset, which you can use to hyperparameter tune the combiner classifier the same way as the individual subsystems. You use the combined prediction .csv file as the data, and the output text file will inform the optimal parameters for the combiner.

The next cell functions similarly to earlier cells, where dictionaries can be changed with optimal parameters from the tuning, and the combiner model provides final predictions for the dataset and creates an ROC curve for comparison of those predictions to other models, such as the neural network provided in my project description.
The final cells are from the development of my project, where one subsystem ROC curve can be created. This code has been modified to show the feature importance from a given subsystem classifier, which is providing interesting results for our lab in other projects, and related to the use of only the SNR feature. This was not originally a part of my project.

Other Notes: All of my hyperparameters from my tuning runs are stored for each subsystem and the combiner in .txt files in the repository, starting with names for each subsystem or the combiner, respectively.

I cannot provide the neural network code I comprate my classifier code to for my poster, seeing as it is a collaboration for the whole Wade Lab. However, if that is of interest please let me know and we can discuss further! It is of a similar pipeline format.

Furthermore, if you would like the LIGO data I used to run the pipeline yourself, I can provide that as well!
