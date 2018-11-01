# Make CNN model
There are some ways to make model.  


# Testing environment
python : 3.6.6  
Packages : Please look requirements.txt  
OS : macOS High Sierra 10.13.2
I strongly recommend you to use GPU. Or you should use google colabolatory.


# Setup
## 1) Set directory like below.  
But you need not to prepare models/\*.  
And If want to use processed train_data, please put like XY_224.txt and need not 'train_data' directory. But you need train_data when you do't use processed train_data.    
.  
├── README.md  
├── demo  
│   └── **predict_mov.py**  
├── **make_cnn.py**  
├── Models  
│   ├── model  
│   │   ├── logs   
│   │   │   └── events.out.tfevents...
│   │   └── model.h5  
│   └── ...  
└── train_data  
    ├── Alkaline  
    │   ├── 00000_0.jpg  
    │   └── ...  
    ├── LIION  
    │   ├── 00001_0.jpg  
    │   └── ...  
    ├── NICD  
    │   ├── 00002_0.jpg  
    │   └── ...  
    ├── NIMH  
    │   ├── 00003_0.jpg  
    │   └── ...  
    └── XY_224.txt

## 2) Install packages  
```
pip install -r requirements.txt   
```

# Description
## make_cnn.py
You can make cnn model.
  - Make own structure network
  - Use Fine-tuning
	 - VGG16  
	 - ResNet50  
	 - InceptionV3  
   - Xception  

- output
It is debug mode, but example of output.
![output image](https://github.com/emi-cd/learn-CNN/blob/readme/imgs/output.png?raw=true)

- Argment
  - '-F', '--fine' : Use fine-tuning. And you have to choice origin model. Please use '-M'.
  - '-M', '--model' : You can choose pretrained model when you use fine-tuning option. Choose from ['VGG', 'RN', 'I', 'X']. VGG is VGG16, RN is ResNet50, I is InceptionV3 and X is Xception.
  - '--file' : If you want to use processed data. Please enter the name of data file. It should be used joblib to compress.
  - '-D', '--traindata' : Path to the traindata. That directory should have label name. Default is './train_data'.
  - '-N', '--name' : The neme of model. Default is 'model'.
  - '--debug' : Debug mode. Show more information when it running.


## predict_mov.py
Demonstrate using by CNN. It shows posibility.  
If you want to stop this program, please enter 'q'.
[![demo mov](https://github.com/emi-cd/make-CNN/blob/readme/imgs/demo.png?raw=true)](http://www.youtube.com/watch?v=jjC69wbVxvs)


# Compare models
I compared the accuracy with my test data. Also tried some image size and optimizer. If you want to see more details, please see [this page](https://docs.google.com/document/d/14dQYAU1SCiJdCCIgX4ubmGzfGVARKKmA1mqhffvajSY/edit?usp=sharing).
- Own structure
  - IMAGE_SIZE : 224, Optimizer : ‘SGD’
    - Evaluation : [3.9057637075208785e-05, 1.0]
  - IMAGE_SIZE : 224, Optimizer : optimizers.RMSprop(lr=1e-4)
    - Evaluation : *
  - IMAGE_SIZE : 150, Optimizer : optimizers.RMSprop(lr=1e-4)
      - Evaluation : [1.5543963376571541e-07, 1.0]
- Using VGG16
  - IMAGE_SIZE : 224, Optimizer : ‘SGD’
    - Evaluation :  [1.3551500213746776e-05, 1.0]
  - IMAGE_SIZE : 224, Optimizer : optimizers.RMSprop(lr=1e-4)
    - Evaluation : [1.1920930376163597e-07, 1.0]
  - IMAGE_SIZE : 150, Optimizer : optimizers.RMSprop(lr=1e-4)
      - Evaluation : [3.131041632835096e-05, 1.0]
- Using ResNet50
  - IMAGE_SIZE : 224, Optimizer : ‘SGD’
    - Evaluation : [0.0010371402929783525, 1.0]
  - IMAGE_SIZE : 224, Optimizer : optimizers.RMSprop(lr=1e-4) ★
    - Evaluation :  [1.823675346285574e-05, 1.0]
  - IMAGE_SIZE : 150, Optimizer : optimizers.RMSprop(lr=1e-4)
      - Evaluation : [0.0002734722340178183, 1.0]
- Using InceptionV3
  - IMAGE_SIZE : 224, Optimizer : ‘SGD’
    - Evaluation : [0.00038484839238020973, 1.0]
  - IMAGE_SIZE : 224, Optimizer : optimizers.RMSprop(lr=1e-4)
    - Evaluation : [8.236820781932158, 0.4889705882352941]
  - IMAGE_SIZE : 150, Optimizer : ‘SGD’
      - Evaluation :[1.244686820927789, 0.9227941176470589]
- Using Xception
  - IMAGE_SIZE : 224, Optimizer : ‘SGD’
    - Evaluation :  [0.0003328676667639657, 1.0]
  - IMAGE_SIZE : 224, Optimizer : optimizers.RMSprop(lr=1e-4)
    - Evaluation : [0.026546953796775047, 0.9926470588235294]
  - IMAGE_SIZE : 150, Optimizer : ‘SGD’
      - Evaluation : [0.05925784501090865, 0.9963235294117647]


# Usage
## Main flow
- Make own network
```
python make_cnn.py
```

- Fine-tuning
```
python make_cnn.py -F -M VGG
```
```
python make_cnn.py -F -M RN
```
```
python make_cnn.py -F -M I
```
```
python make_cnn.py -F -M X
```
