%CAD
%Tajwar, Eze
%Feature extraction of mass lesions

%Select the desired network
network=1; 
%0 = googlenet
%1 = resnet50
%2 = alexnet
%3 = vgg16 
%4 = vgg19

%reshaping images based on the input size of the pretrained model
if (network==0||network==1||network==3||network==4)
    size_patch = 224;
else 
    size_patch =227;
end


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

%Augmenting the training set
imageAugmenter = imageDataAugmenter('RandRotation',[-30 30],'RandXScale',[0.2 0.2],'RandYScale',[0.2 0.2],'RandXShear',[0.2 0.2],'RandYShear',[0.2 0.2],'RandYTranslation',[0.2 0.2],'RandXTranslation',[0.2 0.2])
datasource = augmentedImageSource([224,224],trainingImages,'DataAugmentation',imageAugmenter)

numTrainImages = numel(trainingImages.Labels);

%Choosing the layer from which the features will be extracted
if(network ==0)
net = googlenet;
lgraph = layerGraph(net);
featureLayer = 'pool5-7x7_s1';
str = 'Googlenet_trained';
batch=32;
end

if(network ==1)
net = resnet50;
lgraph = layerGraph(net);
featureLayer = 'fc1000';
str = 'ResNet50';
batch=12;
end

if(network ==2)
net = alexnet;
lgraph = net.Layers;
featureLayer = 'fc7';
str = 'AlexNet';
batch=32;
end

if(network ==3)
net = vgg16;
lgraph = net.Layers;
featureLayer = 'fc7';
str = 'VGG16Net';
batch=32;
end

if(network ==4)
net = vgg19;
lgraph = net.Layers;
featureLayer = 'fc7';
str = 'VGG19Net';
batch=12;
end

%Extracting features from the chosen layer
trainingFeatures = activations(net, trainingImages, featureLayer, ...
    'MiniBatchSize', batch);

testFeatures = activations(net, validationImages, featureLayer, 'MiniBatchSize',batch);

testLabels = validationImages.Labels;
trainingLabels = trainingImages.Labels;

if(network==0 || network ==1)
trainingFeatures=squeeze(trainingFeatures(1,1,:,:));
testFeatures=squeeze(testFeatures(1,1,:,:));
end

%Saving the features
save(str,'trainingFeatures','trainingLabels','testFeatures','testLabels')

