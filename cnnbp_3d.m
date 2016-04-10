function net = cnnbp_3d(net, y)
    n = numel(net.layers);

    %   error
    net.e = net.o - y;
    %  loss function
    net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);

    %%  backprop deltas
    net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1}{1});
    fvnum = sa(1) * sa(2)* sa(3);
    for j = 1 : numel(net.layers{n}.a{1})
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a{1})
                %-----------my code-----------
                if any(size(net.layers{l}.a{1}{j}) ~= 2*size(net.layers{l + 1}.d{j}))
                    tmp = (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale net.layers{l + 1}.scale]) / net.layers{l + 1}.scale ^ 2);
                    
                    s_tmp = size(tmp);
                    s = size(net.layers{l}.a{1}{j});
                    for i = 1:numel(s)
                        if s(i)~=s_tmp(i)
                            s_tmp(i)=s_tmp(i)-2;
                        end
                    end
                    tmp = tmp(1:s_tmp(1),1:s_tmp(2),1:s_tmp(3));
                    %tmp = tmp(1:s_tmp(1)-2,1:s_tmp(2)-2,1:s_tmp(3)-2);
                    net.layers{l}.d{j} = net.layers{l}.a{1}{j} .* tmp;
                else
                    net.layers{l}.d{j} = net.layers{l}.a{1}{j} .* (1 - net.layers{l}.a{1}{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale net.layers{l + 1}.scale]) / net.layers{l + 1}.scale ^ 2);
                end
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a{1})
                z = zeros(size(net.layers{l}.a{1}{1}));
                for j = 1 : numel(net.layers{l + 1}.a{1})
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a{1})
                for i = 1 : numel(net.layers{l - 1}.a{1})
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{1}{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
