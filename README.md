# covidControl

## Project Description ##
CovidControl comes as part of the p2m project, which is part of our engineering training. In this project, we were asked to create a script that can take a video and determine whether the people in the frame respect the distanciation and whether they wear their face masks.

## Running the project ## 
To run the project, you need to :
1. clone the repository
2. download the models from this [Named Link](https://drive.google.com/file/d/1Xl4WMVmz665ii00ZLbCt_KhIOQZVX0Oj/view?usp=sharing "link")
3. put the files previously downloaded in the models/ folder
4. run the script with this command :
    `python detect.py --source filename`
if you want to run the script with the webcam, you can run this command
    `python detect.py` or `python detect.py --source 0`
other useful parameters are:
    **--project**  to specify the file name
    **--nosave** allows you to not save images/videos

## Examples ##
Here is an example of videos you can use :
[Named Link](https://drive.google.com/file/d/1plERPVrBuUtFjyxqtLZQ2I9re3cm-AWy/view?usp=sharing "Video 1")
( other videos and an example of the execution will be added )

This link has a small documentation of how to run the script
https://drive.google.com/file/d/1wv7WjO55Pd-93y65jDQse1RJaOSQ3BSH/view?usp=sharing
