# ML-Aimbot-Halo-Infinite
 A machine learning aimbot for Halo Infinite, this is for educational purposes only, if you use this in a game there is a possibility you might get banned!
 
 (Windows Only)



https://user-images.githubusercontent.com/98558514/168407844-8fde6a0c-3609-453d-9350-6bf5bdb00014.mp4



 Requirements:
 - A Nvidia GPU, the better the GPU, the better the aimbot performance.
 
 What it is:
 -  This is a machine learning model that has been trained to detect enemies in halo infinite, it will then move your mouse to any enemies it detects, it functions rather well based on my testing. The AI itself is undetected by anti-cheat (at least according to my research), but the mouse movement libraries may be flagged by the anticheat, so far I have not been banned but there is always a possibility. 
 
 Installation Instructions:
 
 - Download this repository.
 - Install anaconda (https://www.anaconda.com/products/distribution) on your computer.
 - Open anaconda as administrator and cd to the directory of the deploy.py script in this repository.
 - Install pytorch and cuda (https://pytorch.org/get-started)
 - Create a virtual environment using this command ``` conda create -n pytorch-gpu python==3.8```
 - Activate the virtual env using ``` conda activate pytorch-gpu ```
 - Install required packages (I'm making a requirements.txt but in the meantime just have a look through and see what packages are needed and pip install them)
 - IMPORTANT! You must set enemy outlines in halo to the colour pineapple or the model will not detect them!

 How to use:
 
 - Open anaconda as administrator and cd to the directory of the deploy.py script in this repository.
 - Run the command ``` conda activate pytorch-gpu ``` to start up the virtual environment
 - Run the command ``` python deploy.py ``` to start the bot, make sure you have halo infinite open and are preferably about to enter a game.
 - If your aimbot seems to be lagging behind, try reducing your games framerate until the CPS your command line will print gets to 30+ (I am currently running my game at 40fps in order to allow the ai to get 30+ CPS)

 Settings:
 ```aimbot = True # Enables aimbot if True

screenShotWidth = 416 # Width of the detection box
screenShotHeight = 416 # Height of the detection box

lock_distance = 75 # Recommended over 60 (this is the minimum distance away the bot will lock from)

headshot_mode = True # Pulls aim up towards head if True

no_headshot_multiplier = 0.2 # Amount multiplier aim pulls up if headshot mode is false
headshot_multiplier = 0.35 # Amount multiplier aim pulls up if headshot mode is true

videoGameWindowTitle = "Halo Infinite" # The title of your game window

movement_amp = 1 # Recommended between 0.5 and 1.5 (this is the snap speed)```

 Future improvements:
 - Add support for using Arduino Leonardo + USB Host Shield to spoof mouse inputs and evade possible anticheat detection on mouse movement.

 Credits:
 - Thanks to the https://rootkit.org discord community for helping me out with parts of this! 
