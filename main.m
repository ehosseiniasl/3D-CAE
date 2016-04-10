%%  Pretrainineg Three Stacked layer of 3D-CAE on Brain MRI
% Ehsan Hosseini-Asl
% University of Louisville
% December 15, 2015

clear all; close all; clc;
addpath('util')

%% load 3D Brain MRI;

load('3d_MRI.mat')

x = cell(1, 1);
x{1}{1} = volume;
x{1}{2} = volume;
y = [0,1;
     1,0];
%% Building 3 layers of Stacked 3D-CAE

filter_No = 5;
filter_size = [3 3 3];

noise = 0.0;

scae = {
    struct('outputmaps', filter_No, 'inputkernel', filter_size, 'outputkernel', filter_size, 'scale', [2 2 2], 'sigma', 0.1, 'momentum', 0.9, 'noise', noise)
    struct('outputmaps', filter_No, 'inputkernel', filter_size, 'outputkernel', filter_size, 'scale', [2 2 2], 'sigma', 0.1, 'momentum', 0.9, 'noise', noise)
    struct('outputmaps', filter_No, 'inputkernel', filter_size, 'outputkernel', filter_size, 'scale', [2 2 2], 'sigma', 0.1, 'momentum', 0.9, 'noise', noise)
};

% opts.rounds     =   20;
opts.rounds     =   1;
opts.epoc       =    1;
opts.batchsize  =    1;
opts.trainsize  =   2;
opts.alpha      = 0.01;
opts.ddinterval =   10;
opts.ddhist     =  0.5;
scae = scaesetup_3d(scae, x{1}{1}, opts);

dbstop if error

%% Training 3D-SCAE

scae = scaetrain_3d(scae, x, opts);

%% initialize CNN

addpath('/Users/Ehsan/Dropbox/Codes/Deep Learning/DeepLearnToolbox/CNN')

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', filter_No, 'kernelsize', filter_size(1)) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', filter_No, 'kernelsize', filter_size(1)) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    struct('type', 'c', 'outputmaps', filter_No, 'kernelsize', filter_size(1)) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

                
opts.alpha = 0.001;
opts.batchsize = 1;
opts.numepochs = 10;

cnn = cnnsetup_3d(cnn, x{1}, y,scae);

%%
% mask = x{1}{1};
% mask(mask~=0)=1;
% train_mask{1} = mask;

cnn = cnntrain_3d(cnn, x{1}, y, opts);

%%
[er, bad, output] = cnntest_3d(cnn, x, y);
