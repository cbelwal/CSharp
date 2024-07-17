The Dasmic Machine Learning Library has been written purely in C# and does not use any C++ routines. Core level parallelization is achieved with .NET's Task Parallel Library (TPL). Plans were in place to add C++ routines to leverage CUDA. 

The main code is under the CodeBase folder which is broken up into 4 main folders:

Algorithms    : Algorithms to train the model and do inference
Common        : Shared libraries to deal with Exceptions, Data Management etc.
Math          : Fundamental Mathetical routines dealing with Matrices and Statistics
UserInterface : Future plan to build a UI similar to the Weka toolset

The Algorithms folder has a separate Library (DLL) project for each of the core ML algorithms. For example, a project will exist for Regression, one separte for NN, DecisionTree, DeepLearning etc. Within each of these there is a Model and Build folder. The Build folder includes routines to train the model, while the Model includes the Classes/data structure to persist the model (for use in Transfer Learning) as well as to run it. That way only the Model files are needed for  inference after deployment.

The library structure follows a strict OO approach. Base classes handle the common routines, and Interfaces provide the uniform structure.

Each routine in Build folder has a corresponding Unit Test, under the top level UnitTests folder which is structured similar to the CodeBase folder. This uniformity is important and should be maintained for consistency.

The DataSets include some of the common data sets but are inclusive of all the data sets the models has been tested in.

NOTE: This is still working in progress, so pieces can be missing. You are welcome to complete the routines *wink* and if there are any questions feel free to email cbelwal@gmail.com

