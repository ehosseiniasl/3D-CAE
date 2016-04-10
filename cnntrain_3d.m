% function net = cnntrain_3d(net, x, y, mask, opts)
function net = cnntrain_3d(net, x, y, opts)
%     m = size(x, 3);
    m = size(x,2);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m);
        for l = 1 : numbatches
%             batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
%             batch_x = x(:, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_x{1} = x{kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)};
%             disp(['subject ' num2str(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize))]);
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
%             batch_mask = mask{kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)};

%             net = cnnff_3d(net, batch_x, batch_mask);
            net = cnnff_3d(net, batch_x);
            if net.o(1) > net.o(2)
                disp(['subject ' num2str(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)) ' is 1']);
            else
                disp(['subject ' num2str(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize)) ' is 2']);
            end
            net = cnnbp_3d(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end
