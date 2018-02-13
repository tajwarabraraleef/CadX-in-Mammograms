%CAD
%Tajwar, Eze
%Feature extraction of test set
network=0; 
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


testingImages = imageDatastore('rot\test\',...
                        'IncludeSubfolders',true,...
                        'LabelSource','foldernames');
%Resize the images according to the input of the alexnet
testingImages.ReadFcn = @(loc)repmat(imresize(imread(loc),[size_patch size_patch]), 1, 1, 3);



numTestImages = numel(testingImages.Labels);
numClasses = numel(categories(testingImages.Labels))


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

testFeatures = activations(net, testingImages, featureLayer, ...
    'MiniBatchSize', batch);

if(network==0 || network ==1)
testFeatures=squeeze(testFeatures(1,1,:,:));
end

%Saving extracted features of the test set
save([str 'test'],'testFeatures')

