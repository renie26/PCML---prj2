%  five-fold cross validation, use CNN features with SVM

clearvars;

% Load features and labels of training data
load train/train.mat;

%% -- Example: five-fold cross validation, use HOG features
fprintf('Splitting into train/test..\n');

% k-fold cross validation
N = length(train.y);
K=5;
idx = randperm(N);
Nk=floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end


Tr = [];
Te = [];

log2c = -1:3;
log2g = -15:-8;
g = 2.^log2g;
c = 2.^log2c;

cLen = length(c);
gLen = length(g);

BerSVMrbf = zeros(gLen);

% for i = 1:cLen
     %for j = 1:gLen
        BerSVMCNNrbf = [];
        for k = 1:K
            % get k'th subgroup in test, others in train
            Te.idxs = idxCV(k,:);
            Tr.idxs = idxCV([1:k-1 k+1:end],:);
            Tr.idxs = Tr.idxs(:);
            Te.y = train.y(Te.idxs);
            Te.X = train.X_cnn(Te.idxs,:);
            Tr.y = train.y(Tr.idxs);
            Tr.X = train.X_cnn(Tr.idxs,:);
            
            fprintf('Training SVM..\n');
            
            %addpath(genpath('where/the/libsvm/'));
            
            % normalize data
            [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
            
            Te.normX = normalize(Te.X, mu, sigma);  % normalize test data
            
            
            % svm RBF kernel
            Tr.y = double(Tr.y);
            Tr.normX = double(Tr.normX);
            
           %cmd = ['-c ', num2str(1), ' -g ', num2str(g(j)), '-h 0'];
            
            model = svmtrain(Tr.y, Tr.normX);
            % SVM linear kernel
            % model2=svmtrain(Tr.y,Tr.normX,'-t 0');
            
            fprintf('begin to predict\n');
            Te.y = double(Te.y);
            Te.normX = double(Te.normX);
            predictlabel = svmpredict(Te.y,Te.normX, model);
            %predictlabel = predict(model,Te.normX);
            
            %predictlabel_linear=svmpredict(Te.y,Te.normX, model2);
            PredSVMCNN(:,k)=predictlabel;
            
            BerSVMCNNrbf = [BerSVMCNNrbf,BER(4, predictlabel, Te.y)];
            
            fprintf('\nTesting error: %.2f%%\n\n', BerSVMCNNrbf * 100 );
            
        end
            %BerSVMrbf(j) = mean(BerSVMCNNrbf);
        
%     end
% end
%%
save('result/BerSVMCNNrbf.mat', 'BerSVMCNNrbf');