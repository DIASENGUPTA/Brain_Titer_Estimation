## Introduction
Brain Titer prediction is usually intrusive and only possible after medical inspection. This is an attempt to perform prediction of few of those titers using Deep Learning algorithms. 

## Brain Abeta Prediction
This attempts to improve performance of Abeta prediction by using Inception network and a VGG-inspired lighter convolution based network as backbone of GlobalLocal Transformer.

Train the models
```
cd Brain\ Abeta\ Estimation/Abeta\ train
python GLT_train_test.py
```

Validate the models
```
cd Brain\ Abeta\ Estimation/Abeta\ train
python GLT_Trial_incep.py
python GLT_Trial_novel.py
```

Additional Regression Data Analysis
```
Brain\ Abeta\ Estimation/Regression\ Try\ 1.ipynb
```

## Brain Age Prediction

This presents detailed code and results for lobe-wise fusion models and Global Local Transformer performance for Brain-Age prediction.

Fusion Network

Preprocessing brain MRI
```
cd Brain\ Age\ Estimation/Fusion\ Network/Preprocessing
python script.py
python flirt.py
python ravens.py
```

Preprocessing lobewise segments
Training Fusion Network
```
cd Brain\ Age\ Estimation/Fusion\ Network/Preprocessing
python mask.py
```

Training Fusion Networks
```
cd Brain\ Age\ Estimation/Fusion\ Network/train
python trainreg.py
```

Testing Fusion Networks:
```
cd Brain\ Age\ Estimation/Fusion\ Network/train
python test.py
```

The above code is to train for the entire brain. Use the lobe-wise training codes for part based training. 

GLT Network:
GLT was also explored for intensity and ravens map of brain MRI scans.

Training Scripts:
```
cd Brain\ Age\ Estimation/Global\ Local\ Transformer/train
python train_test_val.py
python train_test_val_ravens.py
```

Test Scripts:
```
cd Brain\ Age\ Estimation/Global\ Local\ Transformer/train
python test_script.py
python test_script_ravens.py
```

## Acknowledgements

This project is forked from [FiANet](https://github.com/shengfly/FiAnet) and uses [Global Local Transformer](https://github.com/shengfly/global-local-transformer).
