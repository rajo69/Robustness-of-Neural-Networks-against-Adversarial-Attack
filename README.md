# Robust Digit Classification with Custom CNN in MATLAB

## Overview

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for digit classification using MATLAB. The project includes techniques for training a robust network capable of resisting adversarial attacks. 

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Adversarial Attacks](#adversarial-attacks)
- [Training a Robust Network](#training-a-robust-network)
- [Verifying Robustness](#verifying-robustness)
- [Supporting Functions](#supporting-functions)
- [References](#references)

## Getting Started

### Prerequisites
- MATLAB (R2020a or later)
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- Image Processing Toolbox
- Deep Learning Toolbox Model for AlexNet Network support package

### Installation
Clone this repository and navigate to the project directory:
```sh
git clone https://github.com/your_username/robust_digit_classification.git
cd robust_digit_classification
```

### Usage
1. Load and preprocess the training data.
2. Construct and initialize the CNN architecture.
3. Train the network using a custom training loop.
4. Evaluate the network on test data.
5. Generate and test adversarial examples.
6. Train a robust network using adversarial training.
7. Verify the robustness of both normal and robust networks.

## Model Architecture

The CNN architecture used in this project includes:
- An input layer for 28x28 grayscale images.
- Two convolutional layers with ReLU activation.
- A max-pooling layer.
- A fully connected layer.
- A softmax layer for classification.

```matlab
layers = [
    imageInputLayer([28 28 1],Normalization="none")
    convolution2dLayer(3,32,Padding=1)
    reluLayer
    convolution2dLayer(3,64,Padding=1)
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    fullyConnectedLayer(10)
    softmaxLayer];
```

## Training the Model

### Load Training Data
```matlab
rng default
[XTrain,TTrain] = digitTrain4DArrayData;
dsXTrain = arrayDatastore(XTrain,IterationDimension=4);
dsTTrain = arrayDatastore(TTrain);
dsTrain = combine(dsXTrain,dsTTrain);
```

### Initialize Network
```matlab
net = dlnetwork;
net = addLayers(net,layers);
net = initialize(net);
```

### Custom Training Loop
```matlab
numEpochs = 30;
miniBatchSize = 100;
learnRate = 0.01;

mbq = minibatchqueue(dsTrain, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat=["SSCB",""]);

velocity = [];
numObservationsTrain = numel(TTrain);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor(Metrics="Loss", Info="Epoch", XLabel="Iteration");

epoch = 0;
iteration = 0;

while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    shuffle(mbq)
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        [X,T] = next(mbq);
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
        net.State = state;
        [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate);
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100 * iteration / numIterations;
    end
end

save('custom_cnn.mat', 'net');
```

## Evaluating the Model
### Test Network
```matlab
[XTest,TTest] = digitTest4DArrayData;
dsXTest = arrayDatastore(XTest,IterationDimension=4);
dsTTest = arrayDatastore(TTest);
dsTest = combine(dsXTest,dsTTest);

mbqTest = minibatchqueue(dsTest, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");

YPred = modelPredictions(net,mbqTest,classes);
acc = mean(YPred == TTest);
```

## Adversarial Attacks
### Generate Adversarial Examples using Basic Iterative Method (BIM)
```matlab
epsilon = 0.1;
alpha = 0.01;
numAdvIter = 20;

reset(mbqTest)
[XAdv,YPredAdv] = adversarialExamples(net,mbqTest,epsilon,alpha,numAdvIter,classes);
accAdversarial = mean(YPredAdv == TTest);
visualizePredictions(XAdv,YPredAdv,TTest);
```

## Training a Robust Network
### Adversarial Training Loop
```matlab
netRobust = dlnetwork;
netRobust = addLayers(netRobust,layers);
netRobust = initialize(netRobust);

numIter = 1;
initialization = "random";
alpha = 1.25 * epsilon;

monitorRobust = trainingProgressMonitor(Metrics="Loss", Info="Epoch", XLabel="Iteration");

velocity = [];
epoch = 0;
iteration = 0;

while epoch < numEpochs && ~monitorRobust.Stop
    epoch = epoch + 1;
    shuffle(mbq)
    while hasdata(mbq) && ~monitorRobust.Stop
        iteration = iteration + 1;
        [X,T] = next(mbq);
        X = basicIterativeMethod(netRobust,X,T,alpha,epsilon,numIter,initialization);
        [loss,gradients,state] = dlfeval(@modelLoss,netRobust,X,T);
        net.State = state;
        [netRobust,velocity] = sgdmupdate(netRobust,gradients,velocity,learnRate);
        recordMetrics(monitorRobust,iteration,Loss=loss);
        updateInfo(monitorRobust,Epoch=epoch + " of " + numEpochs);
        monitorRobust.Progress = 100 * iteration / numIterations;
    end
end

save('custom_cnn_robust.mat', 'netRobust');
```

## Verifying Robustness
### Load Networks
```matlab
load('custom_cnn.mat');
net = removeLayers(net,"softmax");
net = initialize(net);

load('custom_cnn_robust.mat');
netRobust = removeLayers(netRobust,"softmax");
netRobust = initialize(netRobust);
```

### Verify Network Robustness
```matlab
perturbation = 0.05;
XLower = X - perturbation;
XUpper = X + perturbation;

result = verifyNetworkRobustness(net,XLower,XUpper,T);
resultRobust = verifyNetworkRobustness(netRobust,XLower,XUpper,T);

combineResults = [countcats(result), countcats(resultRobust)];
figure
bar(combineResults)
xticklabels(categories(result))
ylabel("Number of Observations")
legend(["Normal Network","Robust Network"],Location="northwest")
```

## Supporting Functions
### Model Loss Function
```matlab
function [loss,gradients,state] = modelLoss(net,X,T)
    [YPred,state] = forward(net,X);
    loss = crossentropy(YPred,T);
    gradients = dlgradient(loss,net.Learnables);
end
```

### Model Gradients Input Function
```matlab
function gradient = modelGradientsInput(net,X,T)
    T = squeeze(T);
    T = dlarray(T,'CB');
    [YPred] = forward(net,X);
    loss = crossentropy(YPred,T);
    gradient = dlgradient(loss,X);
end
```

### Mini-Batch Preprocessing Function
```matlab
function [X,T] = preprocessMiniBatch(XCell,TCell)
    X = cat(4,XCell{1:end});
    X = single(X);
    T = cat(2,TCell{1:end});
    T = onehotencode(T,1);
end
```

### Model Predictions Function
```matlab
function predictions = modelPredictions(net,mbq,classes)
    predictions = [];
    while hasdata(mbq)
        XTest = next(mbq);
        YPred = predict(net,XTest);
        YPred = onehotdecode(YPred,classes,1)';
        predictions = [predictions; YPred];
    end
end
```

### Adversarial Examples Function
```matlab
function [XAdv,predictions] = adversarialExamples(net,mbq,epsilon,alpha,numIter,classes)
    XAdv = {};
    predictions = [];
    iteration = 0;
    while hasdata(mbq)
        iteration = iteration + 1;
        [X,T] = next(mbq);
        initialization = "zero";
        XAdvMBQ = basicIterativeMethod(net,X,T,alpha,epsilon,numIter,initialization);
        YPred = predict(net,XAdvMBQ);
        YPred = onehotdecode(YPred,classes,1)';
        XAdv{iteration} = X

AdvMBQ;
        predictions = [predictions; YPred];
    end
    XAdv = cat(4,XAdv{:});
end
```

### Basic Iterative Method Function
```matlab
function XAdv = basicIterativeMethod(net,X,T,alpha,epsilon,numIter,initialization)
    if initialization == "zero"
        delta = zeros(size(X),like=X);
    else
        delta = epsilon * (2 * rand(size(X),like=X) - 1);
    end
    for i = 1:numIter
        gradient = dlfeval(@modelGradientsInput,net,X + delta,T);
        delta = delta + alpha * sign(gradient);
        delta(delta > epsilon) = epsilon;
        delta(delta < -epsilon) = -epsilon;
    end
    XAdv = X + delta;
end
```

### Visualize Prediction Results Function
```matlab
function visualizePredictions(XTest,YPred,TTest)
    figure
    height = 4;
    width = 4;
    numImages = height * width;
    indices = randperm(size(XTest,4),numImages);
    XTest = extractdata(XTest);
    XTest = XTest(:,:,:,indices);
    YPred = YPred(indices);
    TTest = TTest(indices);
    for i = 1:(numImages)
        subplot(height,width,i)
        imshow(XTest(:,:,:,i))
        if YPred(i) == TTest(i)
            color = "\color{green}";
        else
            color = "\color{red}";
        end
        title("Prediction: " + color + string(YPred(i)))
    end
end
```

## References
1. Szegedy, Christian, et al. "Intriguing Properties of Neural Networks." Preprint, submitted February 19, 2014. [arXiv:1312.6199](https://arxiv.org/abs/1312.6199).
2. Goodfellow, Ian J., et al. "Explaining and Harnessing Adversarial Examples." Preprint, submitted March 20, 2015. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572).
3. Kurakin, Alexey, et al. "Adversarial Examples in the Physical World." Preprint, submitted February 10, 2017. [arXiv:1607.02533](https://arxiv.org/abs/1607.02533).
4. Madry, Aleksander, et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." Preprint, submitted September 4, 2019. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083).
5. Wong, Eric, et al. "Fast Is Better than Free: Revisiting Adversarial Training." Preprint, submitted January 12, 2020. [arXiv:2001.03994](https://arxiv.org/abs/2001.03994).

This project is developed and maintained by [Rajarshi Nandi](https://github.com/rajo69). For any questions or contributions, feel free to open an issue or submit a pull request.
