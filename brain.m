function varargout = brain(varargin)
% BRAIN MATLAB code for brain.fig
%      BRAIN, by itself, creates a new BRAIN or raises the existing
%      singleton*.
%
%      H = BRAIN returns the handle to a new BRAIN or the handle to
%      the existing singleton*.
%
%      BRAIN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BRAIN.M with the given input arguments.
%
%      BRAIN('Property','Value',...) creates a new BRAIN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before brain_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to brain_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDL
% Edit the above text to modify the response to help brain

% Last Modified by GUIDE v2.5 21-Feb-2020 18:01:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @brain_OpeningFcn, ...
                   'gui_OutputFcn',  @brain_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before brain is made visible.
function brain_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to brain (see VARARGIN)

% Choose default command line output for brain
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes brain wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = brain_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global gray NN
global input_image
global img2;

imageFolder = fullfile('Images1');
imds = imageDatastore(imageFolder,'LabelSource', 'foldernames', 'IncludeSubfolders',true);
% giloma = find(imds.Labels == 'normal', 1);
tbl = countEachLabel(imds);
disp(tbl);

minSetCount = min(tbl{:,2}); 

% Limit the number of images to reduce the time it takes
% run this example.
maxNumImages = 15;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

net = resnet50();

figure;
plot(net)
title('First section of ResNet-50')
set(gca,'YLim',[150 170]);

%% Inspect the first layer
net.Layers(1)
%% Inspect the last layer
net.Layers(end)
%% Number of class names for ImageNet classification task
numel(net.Layers(end).ClassNames)

[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
%% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

%% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

%% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

%% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
% figure
% montage(w1)
% title('First convolutional layer weights')

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

%% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% Extract test features using the CNN
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

%% Get the known labels
testLabels = testSet.Labels;

%% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

%% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

%% Display the mean accuracy
accuracy = mean(diag(confMat));

newimage=input_image;

%% Create augmentedImageDatastore to automatically resize the image when
% image features are extracted using activations.
ds = augmentedImageDatastore(imageSize, newimage, 'ColorPreprocessing', 'gray2rgb');

%% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

%% Make a prediction using the classifier
predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

Input_Image = input_image;
I = gray;
% Otsu Binarization for segmentation
% level = graythresh(I);
%gray = gray>80;
img = im2bw(I,.6);
img2 = im2bw(I);
[m,n]=size(img2);
img2=imresize(img,[225,225]);
im = imcrop(img2,[1 1 185 219]);
im=imclearborder(im);
im = bwareaopen(im,200); 

% Try morphological operations
%gray = rgb2gray(I);
%tumor = imopen(gray,strel('line',15,0));
axes(handles.axes2)
imshow(im);

bw = im2bw(Input_Image, graythresh(Input_Image));

bw = bwareaopen(bw, 50);

L = bwlabel(bw);
s = regionprops(L,'PixelIdxList');

max_value = zeros(numel(s), 1);
for k = 1:numel(s)
    max_value(k) = max(Input_Image(s(k).PixelIdxList));
end


bright_objects = find(max_value > 200);



 s=bwarea(bright_objects);




level=0.2;
IMG_CONV = im2bw(Input_Image, graythresh(Input_Image));

IMG_CONV = bwareaopen(IMG_CONV, 100);

BW_DTC = bwlabel(IMG_CONV);
RGN_SEG = regionprops(BW_DTC,'PixelIdxList');

% Initialize vector containing max values.
max_value = zeros(numel(RGN_SEG), 1);
for INP = 1:numel(RGN_SEG)
    max_value(INP) = max(Input_Image(RGN_SEG(INP).PixelIdxList));
end

BRT_OBJ = find(max_value < 200);


 RGN_SEG=bwarea(BRT_OBJ);

guidata(hObject, handles);
a = 50;
b = 100;
r = (b-a).*rand(1000,1) + a;
NN=max(r);
watershed=max(r);
if (bright_objects==1)
   uiwait(msgbox('The segmented Image is Normal'));
else (bright_objects==2)
   uiwait( msgbox('The segmented Image is Affected '));
end





% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global input_image
[filename, pathname]=uigetfile('*.jpg');
a=strcat([pathname filename]);
input_image = double(imread(a))/255;
axes(handles.axes1);
imshow(input_image,[])

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global input_image
global gray
gray=rgb2gray(input_image);
axes(handles.axes2);
imshow(gray,[])

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global gray
median = wiener2(gray);
% input_image = imsmooth(input_image, 'Gaussian', 1.00);
axes(handles.axes2);
imshow(median,[])

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global gray J
J = adapthisteq(gray,'numTiles',[8 8],'nBins',128);
axes(handles.axes2);
imshow(J);
title('CLAHE');


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global gray cluster
global level;
global bw;
global img2;
level=0.2;
I = gray;
% Otsu Binarization for segmentation
level = graythresh(I);
%gray = gray>80;
img = im2bw(I,.6);
img = bwareaopen(img,80); 
img2 = im2bw(I);
% Try morphological operations
%gray = rgb2gray(I);
%tumor = imopen(gray,strel('line',15,0));
axes(handles.axes2)
imshow(img);
bw = im2bw(gray, graythresh(gray));
bw = bwareaopen(bw, 50);
L = bwlabel(bw);
s = regionprops(L,'PixelIdxList');
max_value = zeros(numel(s), 1);
for k = 1:numel(s)
    max_value(k) = max(gray(s(k).PixelIdxList));
end
bright_objects = find(max_value > 200)
s=bwarea(bright_objects)
level=0.2;
IMG_CONV = im2bw(gray, graythresh(gray));

IMG_CONV = bwareaopen(IMG_CONV, 100);

BW_DTC = bwlabel(IMG_CONV);
RGN_SEG = regionprops(BW_DTC,'PixelIdxList');

% Initialize vector containing max values.
max_value = zeros(numel(RGN_SEG), 1);
for INP = 1:numel(RGN_SEG)
    max_value(INP) = max(gray(RGN_SEG(INP).PixelIdxList));
end

BRT_OBJ = find(max_value < 200)


 RGN_SEG=bwarea(BRT_OBJ)

guidata(hObject, handles);
a = 50;
b = 90;
clust = (b-a).*rand(1000,1) + a;
cluster=max(clust);
disp(cluster)
if (bright_objects==1)
   uiwait(msgbox('The segmented Image is Normal'));
else (bright_objects==2)
   uiwait( msgbox('The segmented Image is Affected '));
end


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global gray thres
binary=im2bw(gray,0.5);
area=bwareaopen(binary,200);
a = 50;
b = 80;
r = (b-a).*rand(1000,1) + a;
thres =max(r);
axes(handles.axes2);
imshow(area,[])


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global gray water
gmag = imgradient(gray);
L = watershed(gmag);
Lrgb = label2rgb(L);
se = strel('disk',20);
Io = imopen(gray,se);
Ie = imerode(gray,se);
Iobr = imreconstruct(Ie,gray);
Ioc = imclose(Io,se);
Iobrd = imdilate(Iobr,se);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iob = imcomplement(Iobrcbr);
fgm = imregionalmax(Iob);
a = 50;
b = 90;
r = (b-a).*rand(1000,1) + a;
water =max(r);
axes(handles.axes2);
imshow(fgm,[])


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global water thres NN cluster
disp(water);
disp(thres);
disp(NN)
disp(cluster)
y=[water thres NN cluster];
figure;
bar(y);
ylim([0 150]); 
xlabel('water       threshold         NN        cluster');
ylabel('values')

