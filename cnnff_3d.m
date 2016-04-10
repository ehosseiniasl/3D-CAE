% function net = cnnff_3d(net, x, mask)
function net = cnnff_3d(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
%     inputmaps = 1;
    inputmaps = numel(x);

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{1}{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.a{1}{j} = sigm(z + net.layers{l}.b{j});
                
                %------my code--------------------------
                %remove the background from outputmap
%                 if l == 2
%                     z_mask = convn(mask, ones(size(net.layers{l}.k{i}{j},1),size(net.layers{l}.k{i}{j},2),size(net.layers{l}.k{i}{j},3)), 'valid');
%                     z_mask(z_mask~=0) = 1;
%                     net.layers{l}.a{1}{j} = net.layers{l}.a{1}{j} .* z_mask;
%                 end
                
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{1}{j}, ones(net.layers{l}.scale,net.layers{l}.scale,net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{1}{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end);
                mapsize = size(net.layers{l}.a{1}{j});
                for i = 1:numel(mapsize)
                    if mod(mapsize(i),2) ~= 0 % zero-padding the feature map
                        if i == 1
                            net.layers{l}.a{1}{j} = padarray(net.layers{l}.a{1}{j},[1,0,0],'post');
                        elseif i == 2
                            net.layers{l}.a{1}{j} = padarray(net.layers{l}.a{1}{j},[0,1,0],'post');
                        else
                            net.layers{l}.a{1}{j} = padarray(net.layers{l}.a{1}{j},[0,0,1],'post');
                        end
                    end
                end
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a{1})
        sa = size(net.layers{n}.a{1}{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{1}{j}, sa(1) * sa(2)* sa(3), 1)];
    end
    %  feedforward into output perceptrons
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
    
%     prob = exp(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
% 
%     net.o = prob./repmat(sum(prob),2,1);

end
