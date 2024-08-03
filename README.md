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

This presents detailed code and results for fusion models and lobe-wise Global Local Transformer performance for Brain-Age prediction.







## Acknowledgements

This project is forked from [FiANet](https://github.com/shengfly/FiAnet) and uses [Global Local Transformer](https://github.com/shengfly/global-local-transformer).
