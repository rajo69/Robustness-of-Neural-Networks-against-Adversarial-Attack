%Load Training Data
rng default
[XTrain,TTrain] = digitTrain4DArrayData;

dsXTrain = arrayDatastore(XTrain,IterationDimension=4);
dsTTrain = arrayDatastore(TTrain);

dsTrain = combine(dsXTrain,dsTTrain);

%Extract the class names
classes = categories(TTrain);

%Construct Network Architecture
net = dlnetwork;

%Specify the layers of the classification branch and add them to the network
layers = [
    imageInputLayer([28 28 1],Normalization="none")
    convolution2dLayer(3,32,Padding=1)
    reluLayer
    convolution2dLayer(3,64,Padding=1)
    reluLayer
    maxPooling2dLayer(2,Stride=2)
    fullyConnectedLayer(10)
    softmaxLayer];

net = addLayers(net,layers);

%Initialize the network
net = initialize(net);

%Train Network
%Train the network using a custom training loop. 
%Specify the training options. Train for 30 epochs with a mini-batch size of 100 and a learning rate of 0.01.
numEpochs = 30;
miniBatchSize = 100;
learnRate = 0.01;
Create a minibatchqueue object that processes and manages mini-batches of images during training.
mbq = minibatchqueue(dsTrain, ...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB",""]);
%Initialize the velocity parameter for the SGDM solver.
velocity = [];
%Calculate the total number of iterations for the training progress monitor.
numObservationsTrain = numel(TTrain);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
%Initialize the TrainingProgressMonitor object.
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");
Train the network using a custom training loop. For each epoch, shuffle the data and loop over mini-batches of data.

epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq)

    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration +1;

        % Read mini-batch of data.
        [X,T] = next(mbq);

        % Evaluate the model loss, gradients, and state.
        [loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
        net.State = state;

        % Update the network parameters using the SGDM optimizer.
        [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate);

        % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100 * iteration/numIterations;
    end
end

%Save Network
save('custom_cnn.mat', 'net');

%Test Network
%Create a minibatchqueue object containing the test data.
[XTest,TTest] = digitTest4DArrayData;

dsXTest = arrayDatastore(XTest,IterationDimension=4);
dsTTest = arrayDatastore(TTest);

dsTest = combine(dsXTest,dsTTest);

mbqTest = minibatchqueue(dsTest, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");
%Predict the classes of the test data using the trained network and the modelPredictions function defined at the end of this example.
YPred = modelPredictions(net,mbqTest,classes);
acc = mean(YPred == TTest)
%The network accuracy is very high. 

%Test Network with Adversarial Inputs
%Create adversarial examples using the BIM. Set epsilon to 0.1.
epsilon = 0.1
%Define the step size alpha and the number of iterations.
alpha = 0.01;
numAdvIter = 20;
%Use the adversarialExamples function (defined at the end of this example) to compute adversarial examples using the BIM on the test data set.
reset(mbqTest)
[XAdv,YPredAdv] = adversarialExamples(net,mbqTest,epsilon,alpha,numAdvIter,classes);

%Compute the accuracy of the network on the adversarial example data.
accAdversarial = mean(YPredAdv == TTest)
%Plot the results.
visualizePredictions(XAdv,YPredAdv,TTest);
%You can see that the accuracy is severely degraded by the BIM, even though the image perturbation is hardly visible.

%Train Robust Network
%Train on FSGM examples.
netRobust = dlnetwork;
netRobust = addLayers(netRobust,layers);
netRobust = initialize(netRobust);
%Set the number of iterations to 10 to convert from FSGM to PGD.
numIter = 1;
initialization = "random";
alpha = 1.25*epsilon;
%Initialize the TrainingProgressMonitor object.
monitorRobust = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");
%Train the robust network using a custom training loop and the same training options as previously defined.
velocity = [];
epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs && ~monitorRobust.Stop
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq)

    % Loop over mini-batches.
    while hasdata(mbq) && ~monitorRobust.Stop
        iteration = iteration + 1;

        % Read mini-batch of data.
        [X,T] = next(mbq);

        % Apply adversarial perturbations to the data.
        X = basicIterativeMethod(netRobust,X,T,alpha,epsilon, ...
            numIter,initialization);

        % Evaluate the model loss, gradients, and state.
        [loss,gradients,state] = dlfeval(@modelLoss,netRobust,X,T);
        net.State = state;

        % Update the network parameters using the SGDM optimizer.
        [netRobust,velocity] = sgdmupdate(netRobust,gradients,velocity,learnRate);

        % Update the training progress monitor.
        recordMetrics(monitorRobust,iteration,Loss=loss);
        updateInfo(monitorRobust,Epoch=epoch + " of " + numEpochs);
        monitorRobust.Progress = 100 * iteration/numIterations;
    end
end 

%Save Robust Network
save('custom_cnn_robust.mat', 'netRobust');

%Test Robust Network
%Calculate the accuracy of the robust network on the digits test data.
reset(mbqTest)
YPred = modelPredictions(netRobust,mbqTest,classes);
accRobust = mean(YPred == TTest)
%Compute the adversarial accuracy.
reset(mbqTest)
[XAdv,YPredAdv] = adversarialExamples(netRobust,mbqTest,epsilon,alpha,numAdvIter,classes);
accRobustAdv = mean(YPredAdv == TTest)

%The adversarial accuracy of the robust network is much better than that of the original network.
%Verify Robustness
Load both the networks
load('custom_cnn_robust.mat');

netRobust = removeLayers(netRobust,"softmax");
netRobust = initialize(netRobust);

load('custom_cnn.mat');

net = removeLayers(net,"softmax");
net = initialize(net);
%Verification of the whole test set can take a long time. Use a subset of the test data for verification.
numObservations = numel(TTest);
numToVerify = 200;

idx = randi(numObservations,numToVerify,1);
X = XTest(:,:,:,idx);
T = TTest(idx);
%Convert the test images to a dlarray object with the data format "SSCB" (spatial, spatial, channel, batch), which represents image data.
X = dlarray(X,"SSCB");
%Verify Network Robustness
%To verify the adversarial robustness of a deep learning network, use the verifyNetworkRobustness function. The verifyNetworkRobustness function requires the Deep Learning Toolbox™ Verification Library support package.
%To verify network robustness, the verifyNetworkRobustness function checks that, for all inputs between the specified input bounds, there does not exist an adversarial example. The absence of an adversarial example means that, for all images within the input set defined by the lower and upper input bounds, the predicted class label matches the specified label (usually the true class label).
%For each set of input lower and upper bounds, the function returns one of these values:
%"verified" — The network is robust to adversarial inputs between the specified bounds.
%"violated" — The network is not robust to adversarial inputs between the specified bounds.
%"unproven" — The function cannot prove whether the network is robust to adversarial inputs between the specified bounds.
%Create lower and upper bounds for each of the test images. Verify the network robustness to an input perturbation between –0.05 and 0.05 for each pixel.
perturbation = 0.05;

XLower = X - perturbation;
XUpper = X + perturbation;

%Verify the network robustness for the normal network.

net = initialize(net);

result = verifyNetworkRobustness(net,XLower,XUpper,T);
summary(result)
%Plot the result
figure
bar(countcats(result))
xticklabels(categories(result))
ylabel("Number of Observations")
%Verify the network robustness for the robust network.
resultRobust = verifyNetworkRobustness(netRobust,XLower,XUpper,T);
summary(resultRobust)
%Compare the results from the two networks.
combineResults = [countcats(result),countcats(resultRobust)];
figure
bar(combineResults)
xticklabels(categories(result))
ylabel("Number of Observations")
legend(["Normal Network","Robust Network"],Location="northwest")

%Supporting Functions
%Model Loss Function
%The modelLoss function takes as input a dlnetwork object net and a mini-batch of input data X with corresponding labels T and returns the loss, the gradients of the loss with respect to the learnable parameters in net, and the network state. To compute the gradients automatically, use the dlgradient function.
function [loss,gradients,state] = modelLoss(net,X,T)

[YPred,state] = forward(net,X);

loss = crossentropy(YPred,T);
gradients = dlgradient(loss,net.Learnables);

end
%Input Gradients Function
%The modelGradientsInput function takes as input a dlnetwork object net and a mini-batch of input data X with corresponding labels T and returns the gradients of the loss with respect to the input data X.
function gradient = modelGradientsInput(net,X,T)

T = squeeze(T);
T = dlarray(T,'CB');

[YPred] = forward(net,X);

loss = crossentropy(YPred,T);
gradient = dlgradient(loss,X);

end
%Mini-Batch Preprocessing Function
%The preprocessMiniBatch function preprocesses a mini-batch of predictors and labels using the following steps:
%Extract the image data from the incoming cell array and concatenate into a four-dimensional array.
%Extract the label data from the incoming cell array and concatenate into a categorical array along the second dimension.
%One-hot encode the categorical labels into numeric arrays. Encoding into the first dimension produces an encoded array that matches the shape of the network output.
function [X,T] = preprocessMiniBatch(XCell,TCell)

% Concatenate.
X = cat(4,XCell{1:end});

X = single(X);

% Extract label data from the cell and concatenate.
T = cat(2,TCell{1:end});

% One-hot encode labels.
T = onehotencode(T,1);

end
%Model Predictions Function
%The modelPredictions function takes as input a dlnetwork object net, a minibatchqueue of input data mbq, and the network classes, and computes the model predictions by iterating over all data in the minibatchqueue object. The function uses the onehotdecode function to find the predicted class with the highest score.
function predictions = modelPredictions(net,mbq,classes)

predictions = [];

while hasdata(mbq)

    XTest = next(mbq);
    YPred = predict(net,XTest);

    YPred = onehotdecode(YPred,classes,1)';

    predictions = [predictions; YPred];
end

end
%Adversarial Examples Function
%Generate adversarial examples for a minibatchqueue object using the basic iterative method (BIM) and predict the class of the adversarial examples using the trained network net.
function [XAdv,predictions] = adversarialExamples(net,mbq,epsilon,alpha,numIter,classes)

XAdv = {};
predictions = [];
iteration = 0;

% Generate adversarial images for each mini-batch.
while hasdata(mbq)

    iteration = iteration +1;
    [X,T] = next(mbq);

    initialization = "zero";

    % Generate adversarial images.
    XAdvMBQ = basicIterativeMethod(net,X,T,alpha,epsilon, ...
        numIter,initialization);

    % Predict the class of the adversarial images.
    YPred = predict(net,XAdvMBQ);
    YPred = onehotdecode(YPred,classes,1)';

    XAdv{iteration} = XAdvMBQ;
    predictions = [predictions; YPred];
end

% Concatenate.
XAdv = cat(4,XAdv{:});

end
%Basic Iterative Method Function
%Generate adversarial examples using the basic iterative method (BIM). This method runs for multiple iterations with a threshold at the end of each iteration to ensure that the entries do not exceed epsilon. When numIter is set to 1, this is equivalent to using the fast gradient sign method (FGSM).
function XAdv = basicIterativeMethod(net,X,T,alpha,epsilon,numIter,initialization)

% Initialize the perturbation.
if initialization == "zero"
    delta = zeros(size(X),like=X);
else
    delta = epsilon*(2*rand(size(X),like=X) - 1);
end

for i = 1:numIter

    % Apply adversarial perturbations to the data.
    gradient = dlfeval(@modelGradientsInput,net,X+delta,T);
    delta = delta + alpha*sign(gradient);
    delta(delta > epsilon) = epsilon;
    delta(delta < -epsilon) = -epsilon;
end

XAdv = X + delta;

end
%Visualize Prediction Results Function
%Visualize images along with their predicted classes. Correct predictions use green text. Incorrect predictions use red text.
function visualizePredictions(XTest,YPred,TTest)

figure
height = 4;
width = 4;
numImages = height*width;

% Select random images from the data.
indices = randperm(size(XTest,4),numImages);

XTest = extractdata(XTest);
XTest = XTest(:,:,:,indices);
YPred = YPred(indices);
TTest = TTest(indices);

% Plot images with the predicted label.
for i = 1:(numImages)
    subplot(height,width,i)
    imshow(XTest(:,:,:,i))

    % If the prediction is correct, use green. If the prediction is false,
    % use red.
    if YPred(i) == TTest(i)
        color = "\color{green}";
    else
        color = "\color{red}";
    end
    title("Prediction: " + color + string(YPred(i)))
end

end
References
[1] Szegedy, Christian, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. “Intriguing Properties of Neural Networks.” Preprint, submitted February 19, 2014. https://arxiv.org/abs/1312.6199.
[2] Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. “Explaining and Harnessing Adversarial Examples.” Preprint, submitted March 20, 2015. https://arxiv.org/abs/1412.6572.
[3] Kurakin, Alexey, Ian Goodfellow, and Samy Bengio. “Adversarial Examples in the Physical World.” Preprint, submitted February 10, 2017. https://arxiv.org/abs/1607.02533.
[4] Madry, Aleksander, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. “Towards Deep Learning Models Resistant to Adversarial Attacks.” Preprint, submitted September 4, 2019. https://arxiv.org/abs/1706.06083.
[5] Wong, Eric, Leslie Rice, and J. Zico Kolter. “Fast Is Better than Free: Revisiting Adversarial Training.” Preprint, submitted January 12, 2020. https://arxiv.org/abs/2001.03994.
Copyright 2020-2023 The MathWorks, Inc.
