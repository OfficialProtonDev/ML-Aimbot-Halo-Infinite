# ML-Aimbot-Halo-Infinite
 A machine learning aimbot for Halo Infinite, this is for educational purposes only, if you use this in a game there is a possibility you might get banned!
 
 (Windows Only)
 
[![Watch the video](https://img.youtube.com/vi/sxBfo7TW7Wo/default.jpg)](https://youtu.be/sxBfo7TW7Wo)
 
 Requirements:
 - A Nvidia GPU, the better the GPU, the better the aimbot performance.
 
 What it is:
 -  This is a machine learning model that has been trained to detect enemies in halo infinite, it will then move your mouse to any enemies it detects, it functions rather well based on my testing. The AI itself is undetected by anti-cheat (at least according to my research), but the mouse movement libraries may be flagged by the anticheat, so far I have not been banned but there is always a possibility. 
 
 Installation Instructions:
 
 - Download this repository.
 - Install anaconda (https://www.anaconda.com/products/distribution) on your computer.
 - Open anaconda as administrator and cd to the directory of the deploy.py script in this repository.
 - Install pytorch and cuda (https://pytorch.org/get-started)
 - Create a virtual environment using this command ``` conda create -n pytorch-gpu ```
 - Activate the virtual env using ``` conda activate pytorch-gpu ```
 - Install required packages (I'm making a requirements.txt but in the meantime just have a look through and see what packages are needed and pip install them)
 - IMPORTANT! You must set enemy outlines in halo to the colour pineapple or the model will not detect them!

 How to use:
 
 - Open anaconda as administrator and cd to the directory of the deploy.py script in this repository.
 - Run the command ``` conda activate pytorch-gpu ``` to start up the virtual environment
 - Run the command ``` python deploy.py ``` to start the bot, make sure you have halo infinite open and are preferably about to enter a game.
 - If your aimbot seems to be lagging behind, try reducing your games framerate until the CPS your command line will print gets to 30+ (I am currently running my game at 40fps in order to allow the ai to get 30+ CPS)

 Future improvements:
 - Add support for using Arduino Leonardo + USB Host Shield to spoof mouse inputs and evade possible anticheat detection on mouse movement.

 Credits:
 - Thanks to the https://rootkit.org discord community for helping me out with parts of this! 
