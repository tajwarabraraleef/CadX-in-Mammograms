%CAD
%Tajwar, Eze
%Classification of mass lesions

%Select the desired network
network=5; 
%0 = googlenet
%1 = resnet50
%2 = alexnet
%3 = vgg16 
%4 = vgg19
%5 = alexnet with a deeper fine tuning network

%reshaping images based on the input size of the pretrained model
if (network==0||network==1||network==3||network==4)
    size_patch = 224;
else 
    size_patch =227;
end

%Preparing the training and validation sets
trainingImages = imageDatastore('adapthist\train\',...
                        'IncludeSubfolders',true,...
                        'LabelSource','foldernames');
%Resize the images according to the input of the alexnet
trainingImages.ReadFcn = @(loc)repmat(imresize(imread(loc),[size_patch size_patch]), 1, 1, 3);
validationImages = imageDatastore('adapthist\val\',...
                        'IncludeSubfolders',true,...
                        'LabelSource','foldernames');
%Resize the images according to the input of the alexnet
validationImages.ReadFcn = @(loc)repmat(imresize(imread(loc),[size_patch size_patch]), 1, 1, 3);


numTrainImages = numel(trainingImages.Labels);
numClasses = numel(categories(trainingImages.Labels))


%Removing last few fully connected layers and adding new layers corresding
%to the new number of classes
if(network==0)
net = googlenet;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');
end

if(network ==1)
%Loading the pretrained resnet50
net = resnet50;
lgraph = layerGraph(net);
%Removing the last few fully connected layers
lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});
%Adding new layers that will be fine tuned to fit the two class problem in
%hand
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'avg_pool','fc');
end
    
if(network ==2)
net = alexnet;
net.Layers;
layersTransfer = net.Layers(1:end-3); %taking all layers except the last final three layers
lgraph = [
    layersTransfer
    fullyConnectedLayer(1000,'WeightLearnRateFactor',40,'BiasLearnRateFactor',40)
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',30,'BiasLearnRateFactor',30)
    softmaxLayer
    classificationLayer];
end    
    
if(network ==3)
net = vgg16;
net.Layers;
layersTransfer = net.Layers(1:end-3); %taking all layers except the last final three layers
lgraph = [
    layersTransfer
    fullyConnectedLayer(1000,'WeightLearnRateFactor',40,'BiasLearnRateFactor',40)
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',30,'BiasLearnRateFactor',30)
    softmaxLayer
    classificationLayer];
end    


if(network ==4)
net = vgg19;
net.Layers;
layersTransfer = net.Layers(1:end-3); %taking all layers except the last final three layers
lgraph = [
    layersTransfer
    fullyConnectedLayer(1000,'WeightLearnRateFactor',40,'BiasLearnRateFactor',40)
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',30,'BiasLearnRateFactor',30)
    softmaxLayer
    classificationLayer];
end    

if(network ==5)
net = alexnet;
net.Layers;
layersTransfer = net.Layers(1:end-3); %taking all layers except the last final three layers
lgraph = [
    layersTransfer
    fullyConnectedLayer(1000,'WeightLearnRateFactor',40,'BiasLearnRateFactor',40)
    fullyConnectedLayer(100,'WeightLearnRateFactor',40,'BiasLearnRateFactor',40) 
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',30,'BiasLearnRateFactor',30)
    softmaxLayer
    classificationLayer];
end    

%Augmenting data to increase training set
imageAugmenter = imageDataAugmenter('RandRotation',[-30 30],'RandXScale',[0.2 0.2],'RandYScale',[0.2 0.2],'RandXShear',[0.2 0.2],'RandYShear',[0.2 0.2],'RandYTranslation',[0.2 0.2],'RandXTranslation',[0.2 0.2])
datasource = augmentedImageSource([224,224],trainingImages,'DataAugmentation',imageAugmenter)


%Defining the training parameters of the network
miniBatchSize = 12;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',1,...
    'InitialLearnRate',1e-5,...
    'VerboseFrequency',numIterationsPerEpoch,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch,...
    'LearnRateDropFactor',0.01,...
    'Momentum',0.75,...
    'LearnRateDropPeriod',7,...
    'L2Regularization',0.004);

%Fine tuning the last layers of the network by freezing the previous layers
%and training the new set of layers
netTransfer = trainNetwork(trainingImages,lgraph,options);


%Predicting the labels of the test set
predictedLabels = classify(netTransfer,validationImages,'MiniBatchSize',12);

%Checking accuracy of the results
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels)

%for saving the trained network
save('Classification_net','netTransfer')

%%for generating confusion matrix 
[C,order] = confusionmat(predictedLabels,valLabels)
plotConfMat(C)
