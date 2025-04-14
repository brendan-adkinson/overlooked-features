function indices = cv_indices(n, kfolds)
%{
Function to return cross-validation indices
Each sample will be assigned to one of the k folds (returned as "indices"
array)
%}
randinds=randperm(n);
ksample=floor(n/kfolds);
all_ind=[];
fold_size=floor(n/kfolds);
for idx=1:kfolds
    all_ind=[all_ind; idx*ones(fold_size, 1)];
end
leftover=mod(n, kfolds);
indices=[all_ind; randperm(kfolds, leftover)'];
indices=indices(randperm(length(indices)));
