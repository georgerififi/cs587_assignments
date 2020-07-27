Setup your Python environment to run the assignment code using TensorFlow 1.0 and PYTHON 3.5 !!!!

[Option 1] Use Anaconda 4.2.0 or newer
https://repo.continuum.io/archive/index.html

The preferred approach for installing all the assignment dependencies is to use Anaconda, 
which is a Python distribution that includes many of the most popular Python packages for 
science, math, engineering and data analysis. 

You need to work using Python 3.5 for this assignment.
You also need to install TensorFlow 1.0 in Anaconda using the instructions in the following link
https://www.tensorflow.org/install/
And make sure to verify it is successfully installed and running (Run the Hello world code snippet provided in the installation guide).

So using Anaconda environment (whatever version 4.2.0 or newer) it is easy to create an environment to use with Python3.5.

step.1.
right-click-open terminal inside anaconda3 folder

step.2.
>>cd envs
>>conda create -n assign2 python=3.5
>>activate assign2
>>conda install -n assign2 -c conda-forge tensorflow=1.0
>>conda install -n assign2 -c conda-forge tensorboard=1.10
>>jupyter notebook

step.3.
then move manually the cs587_assignment2 folder inside the env named assign2 

step.4.
navigate through jupyter to envs/assign2/cs587_assignment2

Scikit-learn is already installed in any Anaconda package.

If you have problems i.e jupyter notebook is not initiated because of "module not recognized/ module not found" error
or in case you get errors while executing any of the given python scripts/modules in the code then:


[Option 2] 
Manual install, virtual environment or conda env using Python 3.5 distro installed in your machine: 

If you’d like to (instead of Anaconda) go with a more manual and risky installation route you will likely want to create a virtual environment for the project. 
If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. 

The following link might be helpful http://stackoverflow.com/questions/5506110/is-it-possible-to-install-another-version-of-python-to-virtualenv

Install Python 3.5
https://www.python.org/
https://www.python.org/downloads/release/python-350/

Install scikit-learn
http://scikit-learn.org/stable/install.html

Install TensorFlow 1.0
https://www.tensorflow.org/install/
