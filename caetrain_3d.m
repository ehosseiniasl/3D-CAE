function cae = caetrain_3d(cae, x, opts)
n = cae.inputkernel(1);
cae.rL = [];
%     for m = 1 : opts.rounds
%         tic;
%         disp([num2str(m) '/' num2str(opts.rounds) ' rounds']);

for k = 1:opts.epoc
    
    disp(['---------' num2str(k) '/' num2str(opts.epoc) ' epoc' '---------']);
    fprintf('\n');
    
    for j = 1:opts.trainsize
        %         i1 = randi(numel(x));
        %         l  = randi(size(x{i1}{1},1) - opts.batchsize - n + 1);
        %         x1{1} = double(x{i1}{1}(l : l + opts.batchsize - 1, :, :)) / 255;
        %         x1{1} = x{1}{1};
        i1 = randi(numel(x));
%         x1{1} = x{i1};
        x1{1} = x{j};
        if numel(x1{1}) < 1000
            x1 = x1{1};
        end
        
        %         if n == 1   %Auto Encoder
        %             x2{1} = x1{1};
        x2 = x1;
        %         x2 = x1;
        %         else        %Predictive Encoder
        %             x2{1} = double(x{i1}{1}(l + n : l + n + opts.batchsize - 1, :, :)) / 255;
        %         end
        %  Add noise to input, for denoising stacked autoenoder
        %     x1{1} = x1{1} .* (rand(size(x1{1})) > cae.noise);
        %             x1 = x1 .* (rand(size(x1)) > cae.noise);
%         disp(['subject' num2str(i1)]);
        disp(['subject' num2str(j)]);
        fprintf('\n');
        
        for m = 1 : opts.rounds
            tic;
            disp([num2str(m) '/' num2str(opts.rounds) ' rounds']);
            
            cae = caeup_3d(cae, x1);
            cae = caedown_3d(cae);
            cae = caebp_3d(cae, x2);
            cae = caesdlm_3d(cae, opts, m);
            %         caenumgradcheck(cae,x1,x2);
            cae = caeapplygrads_3d(cae);
            
            if m == 1
                cae.rL(1) = cae.L;
            end
            disp([num2str(cae.L)])
            %         cae.rL(m + 1) = 0.99 * cae.rL(m) + 0.01 * cae.L;
            cae.rL(m + 1) = cae.L;
            %         if cae.sv < 1e-10
            %             disp('Converged');
            %             break;
            %         end
            toc;
        end
        fprintf('\n');
        
    end
    
end
end
