function scae = scaetrain_3d(scae, x, opts)
%TODO: Transform x through scae{1} into new x. Only works for a single PAE.
for i=1:numel(scae)
    disp(['---------SAE' num2str(i) '---------']);
    fprintf('\n');
    
    %         scae{i} = paetrain(scae{i}, x, opts);
    if i == 1
        hidden = x{1};
        scae{i} = caetrain_3d(scae{i}, hidden, opts);
        
        for l = 1:numel(hidden)
            
            x1{1} = hidden{l};
            scae{i} = caeup_3d(scae{i}, x1);
            X = scae{i}.a;
            ll = size(X{1});
            for j = 1:numel(X)
                B=X{j}(scae{i}.M);
                B=B+rand(size(B))*1e-12;
                B=(B.*(B==repmat(max(B,[],1),[size(B,1) 1])));
                b=B(B~=0);
                b=b(1:size(B,2));
                b=reshape(b,ll/2);
                hidden_tmp{j} = b;
                
                % zero-padding----
                for k = 1:3
                    if mod(size(hidden_tmp{j},k),2) ~= 0
                        if k == 1
                            hidden_tmp{j} = padarray(hidden_tmp{j},[1,0,0],'post');
                        elseif k == 2
                            hidden_tmp{j} = padarray(hidden_tmp{j},[0,1,0],'post');
                        else
                            hidden_tmp{j} = padarray(hidden_tmp{j},[0,0,1],'post');
                        end
                        
                    end
                end
                %-----------------
            end
            
            
            
            scae{i}.hid{1}{l} = hidden_tmp;
        end
        
    else
        
        
        %         hidden = scae{i-1}.hid{1}{l};
        hidden = scae{i-1}.hid{1};
        scae{i} = caetrain_3d(scae{i}, scae{i-1}.hid{1}, opts);
        
        %         for l = 1:numel(hidden)
        for l = 1:numel(hidden)
            %     for k = 1:numel(hidden{1})
            x1{1} = hidden{l};
            scae{i} = caeup_3d(scae{i}, x1{1});
            X = scae{i}.a;
            ll = size(X{1});
            for j = 1:numel(X)
                B=X{j}(scae{i}.M);
                B=B+rand(size(B))*1e-12;
                B=(B.*(B==repmat(max(B,[],1),[size(B,1) 1])));
                b=B(B~=0);
                b=b(1:size(B,2));
                b=reshape(b,ll/2);
                hidden_tmp{j} = b;
                
                % zero-padding----
                for k = 1:3
                    if mod(size(hidden_tmp{j},k),2) ~= 0
                        if k == 1
                            hidden_tmp{j} = padarray(hidden_tmp{j},[1,0,0],'post');
                        elseif k == 2
                            hidden_tmp{j} = padarray(hidden_tmp{j},[0,1,0],'post');
                        else
                            hidden_tmp{j} = padarray(hidden_tmp{j},[0,0,1],'post');
                        end
                        
                    end
                end
            %-----------------
            
            end
            scae{i}.hid{1}{l} = hidden_tmp;
            %     end
        end
        %                 X = scae{i}.a;
        %                 ll = size(X{1});
        %                 for j = 1:numel(X)
        %                     B=X{j}(scae{i}.M);
        %                     B=B+rand(size(B))*1e-12;
        %                     B=(B.*(B==repmat(max(B,[],1),[size(B,1) 1])));
        %                     b=B(B~=0);
        %                     b=b(1:size(B,2));
        %                     b=reshape(b,ll/2);
        %                     hidden{j} = b;
        %                 end
        %
        %                 scae{i}.hid = hidden;
        
        %         end
        
    end
    
end
%     scae{1} = caetrain_3d(scae{1}, x, opts);

end