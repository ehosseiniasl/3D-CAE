% function [er, bad, predict] = cnntest_3d(net, x, y, mask)
function [er, bad, predict] = cnntest_3d(net, x, y)
    %  feedforward
    numdata = size(x{1}, 1);
    
    for l = 1 : numdata
%             batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            input{1} = x{1}{l};
            output = y(:,l);
%             input_mask = mask{l};
%             net = cnnff_3d(net, input, input_mask);
            net = cnnff_3d(net, input);
            predict(:,l) = net.o;
    end
    
    %net = cnnff_3d(net, x{1});
    [~, h] = max(predict);
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
end
