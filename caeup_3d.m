function cae = caeup_3d(cae, x)
cae.i = x;

%init temp vars for parrallel processing
pa  = cell(size(cae.a));
pi  = cae.i;
pik = cae.ik;
pb  = cae.b;

for j = 1 : numel(cae.a)
    z = 0;
    for i = 1 : numel(pi)
        z = z + convn(pi{i}, pik{i}{j}, 'full');
    end
    pa{j} = sigm(z + pb{j});
    %         pa{j} = tanh_opt(z + pb{j});
    
    % zero-padding the feature map
%     for i = 1:3
%         if mod(size(pa{j},i),2) ~= 0 
%             if i == 1
%                 pa{j} = padarray(pa{j},[1,0,0],'post');
%             elseif i == 2
%                 pa{j} = padarray(pa{j},[0,1,0],'post');
%             else
%                 pa{j} = padarray(pa{j},[0,0,1],'post');
%             end
%             
%         end
%     end
    %  Max pool.
    if ~isequal(cae.scale, [1 1 1])
        pa{j} = max3d_3D(pa{j}, cae.M);
    end
    
    
    
end
cae.a = pa;

end
