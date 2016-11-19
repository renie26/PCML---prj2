%  five-fold cross validation, use HOG features with SVM

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


% parameter tuning for rbf kernel using the heuristic given in the report
log2c = -2:2:6;
log2cbest = -12;
log2g = -14:2:-6;
g = 2.^log2g;
c = 2.^log2c;

cLen = length(c);
gLen = length(g);

BerSVMrbfAllHog = zeros(gLen,5);

 for i = 1:cLen
 %    for j = 1:gLen
        BerSVMHog = [];
        for k = 1:1
            % get k'th subgroup in test, others in train
            Te.idxs = idxCV(k,:);
            Tr.idxs = idxCV([1:k-1 k+1:end],:);
            Tr.idxs = Tr.idxs(:);
            Te.y = train.y(Te.idxs);
            Te.X = train.X_hog(Te.idxs,:);
            Tr.y = train.y(Tr.idxs);
            Tr.X = train.X_hog(Tr.idxs,:);
            
            fprintf('Training SVM..\n');
            
            %addpath(genpath('where/the/libsvm/'));
            
            % normalize data
            [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
            
            Te.normX = normalize(Te.X, mu, sigma);  % normalize test data
            
            
            % svm RBF kernel
            Tr.y = double(Tr.y);
            Tr.normX = double(Tr.normX);
            
            % set the parameter in search space
            cmd = ['-c ', num2str(c(i)), ' -g ', num2str(g(i))];
            
            model = svmtrain(Tr.y, Tr.normX, cmd);
            % SVM linear kernel
            % model2=svmtrain(Tr.y,Tr.normX,'-t 0');
            
            fprintf('begin to predict\n');
            Te.y = double(Te.y);
            Te.normX = double(Te.normX);
            predictlabel = svmpredict(Te.y,Te.normX, model);

            PredSVMHog(:,k)=predictlabel;
            
            % compute BER
            BerSVMHog = [BerSVMHog,BER(4, predictlabel, Te.y)];
            
            fprintf('\nTesting error: %.2f%%\n\n', BerSVMHog * 100 );
            
        end
            BerSVMrbfAllHog(i,:) = BerSVMHog;
        
     end
% end
%%
save('result/BerSVMrbfAllHog.mat', 'BerSVMrbfAllHog');

%% draw the results of hyperparameter tuning
p = boxplot(BerSVMrbfAllHog');
set(p,'linewidth',3);
ax = gca;
set(gca,'FontSize',24)
title('Tuning Parameter (C, gamma) for SVM with RBF Kernel For HOG Features');
ax.XTickLabel = {'(2^{-2}, 2^{-14})','(2^{-0}, 2^{-12})','(2^{2}, 2^{-10})','(2^{4}, 2^{-8})','(2^{6}, 2^{-6})','fontsize', 32};
xlabel('Parameter Pair (C, gamma)','fontsize', 32);
ylabel('BER','fontsize', 32);
%hLegend = legend(findall(gca,'Tag','Box'), {'2^{-15}','2^{-14}','2^{-13}','2^{-12}','2^{-11}','2^{-10}','2^{-9}','2^{-8}'} ,'fontsize', 30, 'location', 'northeastoutside');
grid on;

%%
p = boxplot(BerSVMLinearAllHog');
set(p,'linewidth',3);
ax = gca;
set(gca,'FontSize',24)
title('Tuning Parameter C for SVM with linear Kernel For HOG Features');
ax.XTickLabel = {'2^{-16}', '2^{-14}','2^{-12}', '2^{-10})','2^{-8}', '2^{-6}','2^{-4}', '2^{-2}','2^{0}', '2^{2}','fontsize', 32};
xlabel('Parameter Pair C','fontsize', 32);
ylabel('BER','fontsize', 32);
%hLegend = legend(findall(gca,'Tag','Box'), {'2^{-15}','2^{-14}','2^{-13}','2^{-12}','2^{-11}','2^{-10}','2^{-9}','2^{-8}'} ,'fontsize', 30, 'location', 'northeastoutside');
grid on;