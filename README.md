# End-To-End-Automatic-Speech-Recognition
A simple speech to text converter.

# About
There are arround 29 options to chose from :
`yes no up down left right on off stop go zero one two three four five six seven eight nine bed bird cat dog happy house marvin sheila tree wow`
  
There are samples of each recording in the sample directoy of the respective option.
  
To test the classifier copy any audio sample in the audio directory and run the `prog.py`.

# Important Note
This has been coded for the ease of linux system's ONLY.
  
PS : One may feel free to spend a day or two making it compatible with any other Operating System.

# File structure for the Project
.
  
├── audio
  
│   └── *.wav          # all test files with extention .wav
  
├── functions
  
│   ├── gen.py	
  
│   └── bl.py
  
├── prog.py      # the main python file for predicting audio samples present in the folder audio
  
├── record.py      # the python script used to record user voice and store it in the audio directory
  
└── MaDaR4           # folder for model, checkpoints and logs related to the classifier

# Requirements
python3.x
  
numpy>=1.13.3
  
scipy>=0.19.1
  
tensorflow-gpu>=1.4.0
  
tqdm

# Getting Started

clone the repository using : `git clone https://github.com/Pranav-Bhaskar/End-To-End-Automatic-Speech-Recognition`
  
After completing all installations.
  
To record your voice use : `python3 record.py`
  
To predict all your recording's use : `python3 prog.py`

# Installation (FULL)
To install all the required packages on your system run the following : `bash inst.sh`

# Installation (Individually)
It is recomended to remove the keyword sudo from the pip commands if you dont want it to install for all the users.

For pyaudio : `sudo apt-get install python3-pyaudio`

For numpy : `sudo python3 -m pip install numpy`

For Scipy : `sudo python3 -m pip install scipy`

# For Tensorflow

Latest CPU(with GPU support) : `sudo python3 -m pip install tensorflow-gpu==1.11`
  
Old CPU(with GPU support) : `sudo python3 -m pip install tensorflow-gpu==1.5`
  
Latest CPU(without GPU support) : `sudo python3 -m pip install tensorflow==1.11`
  
Old CPU(without GPU support) : `sudo python3 -m pip install tensorflow==1.5`

# Test Installation
To test the installation Run : `python3 test.py`
  
In case of errors debug the respective package.
