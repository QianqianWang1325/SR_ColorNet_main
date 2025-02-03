# SR_ColorNet
SR_ColorNet

1.Data preparation
Taking the WorldView II dataset as an example WorldView-II  
    >>--train  
        >>>>--pan  
        >>>>--ms  
    >>--val  
        >>>>--pan  
        >>>>--ms  
    >>--test  
        >>>>--pan  
        >>>>--ms  

2.Train
Put the training/val/test data into the corresponding path in main.py.
# train
python main.py

3.Test
The pre trained model generated during the training process follows the ./log_1/pkl/best.   
# test
[pre-train model](https://pan.baidu.com/s/1ghTCKOnyHOki6U0I0mX1WA?pwd=12SR)  Extract code: 12SR  
python main.py
