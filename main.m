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

%% Building 3 layers of Stacked 3D-CAE

filter_No = 5;
filter_size = [3 3 3];

noise = 0.0;

scae = {
    struct('outputmaps', filter_No, 'inputkernel', filter_size, 'outputkernel', filter_size, 'scale', [2 2 2], 'sigma', 0.1, 'momentum', 0.9, 'noise', noise)
    struct('outputmaps', filter_No, 'inputkernel', filter_size, 'outputkernel', filter_size, 'scale', [2 2 2], 'sigma', 0.1, 'momentum', 0.9, 'noise', noise)
    struct('outputmaps', filter_No, 'inputkernel', filter_size, 'outputkernel', filter_size, 'scale', [2 2 2], 'sigma', 0.1, 'momentum', 0.9, 'noise', noise)
};

opts.rounds     =   20;
opts.epoc       =    1;
opts.batchsize  =    1;
opts.trainsize  =   17;
opts.alpha      = 0.01;
opts.ddinterval =   10;
opts.ddhist     =  0.5;
scae = scaesetup_3d(scae, x{1}{1}, opts);


%% Training 3D-SCAE

scae = scaetrain_3d(scae, x, opts);

%%
