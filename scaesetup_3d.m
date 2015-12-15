function scae = scaesetup_3d(scae, input, opts)
%     x = x{1};
        x{1} = input;
        
    for l = 1 : numel(scae)
        cae = scae{l};
%         ll= [opts.batchsize size(x{1}, 2) size(x{1}, 3)] + cae.inputkernel - 1;
        if l == 1
%             x = input;
            ll= [size(x{1}, 1) size(x{1}, 2) size(x{1}, 3)] + cae.inputkernel - 1;
%             ll= [size(x, 1) size(x, 2) size(x, 3)] + cae.inputkernel - 1;

        else
            x = scae{l-1}.a;
            ll= [size(x{1}, 1)/2 size(x{1}, 2)/2 size(x{1}, 3)/2] + cae.inputkernel - 1;
            
            % make sure mapsize is even
            for i = 1:numel(ll)
                if mod(ll(i),2) ~= 0
                    ll(i) = ll(i) + 1;
                end
            end

        end
            
      
        

        X = zeros(ll);
        cae.M = nbmap(X, cae.scale);
        bounds = cae.outputmaps * prod(cae.inputkernel) + numel(x) * prod(cae.outputkernel);
        for j = 1 : cae.outputmaps   %  activation maps
%             cae.a{j} = zeros(size(x{1}) + cae.inputkernel - 1);
            cae.a{j} = zeros(ll);
%             cae.a{j} = zeros(size(x) + cae.inputkernel - 1);
            for i = 1 : numel(x)    %  input map
%             for i = 1 : opts.batchsize
                cae.ik{i}{j}  = (rand(cae.inputkernel)  - 0.5) * 2 * sqrt(6 / bounds);
                cae.ok{i}{j}  = (rand(cae.outputkernel) - 0.5) * 2 * sqrt(6 / bounds);
                cae.vik{i}{j} = zeros(size(cae.ik{i}{j}));
                cae.vok{i}{j} = zeros(size(cae.ok{i}{j}));
            end
            cae.b{j} = 0;
            cae.vb{j} = zeros(size(cae.b{j}));
        end

        cae.alpha = opts.alpha;

%         cae.i = cell(numel(x), 1);
%         cae.i = cell(numel(input), 1);
        cae.i = cell(numel(x), 1);
        cae.o = cae.i;

        for i = 1 : numel(cae.o)
            cae.c{i}  = 0;
            cae.vc{i} = zeros(size(cae.c{i}));
        end

        ss = cae.outputkernel;

        if l == 1
            cae.edgemask = zeros([size(x{1}, 1) size(x{1}, 2) size(x{1}, 3)]);
        else
            cae.edgemask = zeros([size(x{1}, 1)/2 size(x{1}, 2)/2 size(x{1}, 3)/2]);
        end
        
        for i = 1:3
            if mod(size(cae.edgemask,i),2) ~= 0
                if i == 1
                    cae.edgemask = padarray(cae.edgemask,[1,0,0],'post');
                elseif i == 2
                    cae.edgemask = padarray(cae.edgemask,[0,1,0],'post');
                else
                    cae.edgemask = padarray(cae.edgemask,[0,0,1],'post');
                end
                
            end
        end


        cae.edgemask(ss(1) : end - ss(1) + 1, ...
                     ss(2) : end - ss(2) + 1, ...
                     ss(3) : end - ss(3) + 1) = 1;

        scae{l} = cae;
    end
    
    function B = nbmap(X,n)
        assert(numel(n)==3,'n should have 3 elements (x,y,z) scaling.');
        X = reshape(1:numel(X),size(X,1),size(X,2),size(X,3));
%         B = zeros(size(X,1)/n(1),prod(n),size(X,2)*size(X,3)/prod(n(2:3)));
% B = zeros(prod(n),size(X,1)*size(X,2)*size(X,3)/prod(n(1:3)));
        u=1;
        p=1;
        B = [];
        
        for i = 1:size(X,3)/n(3)
            
            tmp1 = im2col(X(:,:,i*n(3)-1),n(1:2),'distinct');
            tmp2 = im2col(X(:,:,i*n(3)),n(1:2),'distinct');
            
           B = [B, vertcat(tmp1, tmp2)];
        end
%         for m=1:size(X,1)
% %             B(u,(p-1)*prod(n(1:3))+1:p*prod(n(1:3)),:) = im2col(squeeze(X(m,:,:)),n(1:3),'distinct');
%             B(u,(p-1)*prod(n(1:3))+1:p*prod(n(1:3)),:) = im2col(X,n(1:3),'distinct');
% 
%             p=p+1;
%             if(mod(m,n(1))==0)
%                 u=u+1;
%                 p=1;
%             end
%         end

    end
end
