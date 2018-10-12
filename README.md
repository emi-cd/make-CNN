# learn-CNN
Make CNN model for my larning.


# Dependency
python : 3.6.6  
Packages : Please look requirements.txt  


# Setup
1. Set directory like below.  
But you need not to prepare models/\*.  
And If want to use processed train_data, please put like XY_224.txt.  
.  
├── README.md  
├── demo  
│   └── predict_mov.py  
├── make_cnn.py  
├── models  
│   ├── log_model  
│   │   ├── events.out.tfevents...  
│   │   └── ...  
│   ├── log_...  
│   ├── model.h5  
│   ├── ....h5  
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

2. Install packages  
> pip install -r requirements.txt   

# Description
You can make cnn model by 4 ways.
- Make own structure
- Use Fine-tuning
	- VGG16
	- ResNet50
	- InceptionV3


# Usage
## Main flow
- Make own network
> python make_cnn.py

- Fine-tuning
  > python make_cnn.py -F -M VGG

  or
  > python make_cnn.py -F -M RN

  or
  > python make_cnn.py -F -M I
