%  five-fold cross validation, use HOG features with MLP
clearvars;

% Load features and labels of training data
load train/train.mat;

%% -- Example: five-fold cross validation, use CNN features
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
BerNNHog5 = [];
for k = 1:K
    % get k'th subgroup in test, others in train
    Te.idxs = idxCV(k,:);
    Tr.idxs = idxCV([1:k-1 k+1:end],:);
    Tr.idxs = Tr.idxs(:);
    Te.y = train.y(Te.idxs);
    Te.X = train.X_hog(Te.idxs,:);
    Tr.y = train.y(Tr.idxs);
    Tr.X = train.X_hog(Tr.idxs,:);
    
    addpath(genpath('/Users/mifei/Dropbox/EPFL/Courses/PCML/PCMLprj/proj2/DeepLearnToolbox-master/'));

    setSeed(1);

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.X,2) 200 100 4]);
    opts.numepochs =  20;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples
    
    % if == 1 => plots trainin error as the NN is trained
    opts.plot               = 1;
    
    nn.learningRate = 2;
    
    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);
    
    % normalize data
    [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
    
    % prepare labels for NN
    LL = [1*(Tr.y == 1), ...
        1*(Tr.y == 2), ...
        1*(Tr.y == 3), ...
        1*(Tr.y == 4) ];  % first column, p(y=1)
    % second column, p(y=2), etc
    
    [nn, L] = nntrain(nn, Tr.normX, LL, opts);
    
    
    Te.normX = normalize(Te.X, mu, sigma);  % normalize test data
    
    % to get the scores we need to do nnff (feed-forward)
    %  see for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
    nn.testing = 0;
    
    
    % predict on the test set
    nnPred = nn.a{end};
    
    % get the most likely class
    [~,classVote] = max(nnPred,[],2);
    
    PredNNHog(:,k)=classVote;
    
    %Compute BER
    BerNNHog5 = [BerNNHog5,BER(size(nnPred,2), classVote, Te.y)];
    
    fprintf('\nTesting error: %.2f%%\n\n', BerNNHog5 * 100 );
    
end

save('result/BerNNHog5.mat', 'BerNNHog5');

%% For Binary Classification
load train/train.mat;
train.y(find(train.y ~= 4)) = 1;
train.y(find(train.y == 4)) = 2;

Tr = [];
Te = [];
BerBinNNHog2 = [];
for k = 1:K
    % get k'th subgroup in test, others in train
    Te.idxs = idxCV(k,:);
    Tr.idxs = idxCV([1:k-1 k+1:end],:);
    Tr.idxs = Tr.idxs(:);
    Te.y = train.y(Te.idxs);
    Te.X = train.X_hog(Te.idxs,:);
    Tr.y = train.y(Tr.idxs);
    Tr.X = train.X_hog(Tr.idxs,:);
     
    setSeed(1);

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.X,2) 100 10 2]);
    opts.numepochs =  20;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples
    
    % if == 1 => plots trainin error as the NN is trained
    opts.plot               = 1;
    
    nn.learningRate = 2;
    
    % this neural network implementation requires number of samples to be a
    % multiple of batchsize, so we remove some for this to be true.
    numSampToUse = opts.batchsize * floor( size(Tr.X) / opts.batchsize);
    Tr.X = Tr.X(1:numSampToUse,:);
    Tr.y = Tr.y(1:numSampToUse);
    
    % normalize data
    [Tr.normX, mu, sigma] = zscore(Tr.X); % train, get mu and std
    
    % prepare labels for NN
    LL = [1*(Tr.y == 1), ...
        1*(Tr.y == 2)];  
    
    [nn, L] = nntrain(nn, Tr.normX, LL, opts);
    
    
    Te.normX = normalize(Te.X, mu, sigma);  % normalize test data
    
    % to get the scores we need to do nnff (feed-forward)
    %  see for example nnpredict().
    % (This is a weird thing of this toolbox)
    nn.testing = 1;
    nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
    nn.testing = 0;
    
    
    % predict on the test set
    nnPred = nn.a{end};
    
    % get the most likely class
    [~,classVote] = max(nnPred,[],2);
    PredBinNNHog(:,k)=classVote;  
    % Compute BER
    BerBinNNHog2 = [BerBinNNHog2,BER(size(nnPred,2), classVote, Te.y)];
    
    fprintf('\nTesting error: %.2f%%\n\n', BerBinNNHog2 * 100 );
    
end

save('result/BerBinNNHog2.mat', 'BerBinNNHog2');



%% plot the results for applying MLP for HOG features
load('result/BerNNHog.mat');
load('result/BerNNHog2.mat');
load('result/BerNNHog3.mat');
load('result/BerNNHog4.mat');
load('result/BerNNHog5.mat');

%%

p = boxplot([BerNNHog;BerNNHog3;BerNNHog4;BerNNHog6; BerNNHog2; BerNNHog5]');
ax = gca;
set(p,'linewidth',3);
set(gca,'FontSize',24)
title(' MLP using Hog Feature for Multi-class Classification');
ax.XTickLabel = {'One (100)','One (200)','One (500)','One (800)', 'Two (100 10)', 'Two (200 100)','fontsize', 26};
xlabel('Number of Hidden Layers and Size','fontsize', 32);
ylabel('BER','fontsize', 32);
%hLegend = legend(findall(gca,'Tag','Box'), {'2^{-15}','2^{-14}','2^{-13}','2^{-12}','2^{-11}','2^{-10}','2^{-9}','2^{-8}'} ,'fontsize', 30, 'location', 'northeastoutside');
grid on;


%%
load('result/BerBinNNHog.mat');
load('result/BerBinNNHog2.mat');
load('result/BerBinNNHog3.mat');
load('result/BerBinNNHog4.mat');
load('result/BerBinNNHog5.mat');


%%
p = boxplot([BerBinNNHog;BerBinNNHog3;BerBinNNHog4;BerBinNNHog6; BerBinNNHog2; BerBinNNHog5]');
ax = gca;
set(p,'linewidth',3);
set(gca,'FontSize',24)
title(' MLP using Hog Feature for Binary Classification');
ax.XTickLabel = {'One (100)','One (200)','One (500)','One (800)', 'Two (100 10)', 'Two (200 100)','fontsize', 26};
xlabel('Number of Hidden Layers and Size','fontsize', 32);
ylabel('BER','fontsize', 32);
%hLegend = legend(findall(gca,'Tag','Box'), {'2^{-15}','2^{-14}','2^{-13}','2^{-12}','2^{-11}','2^{-10}','2^{-9}','2^{-8}'} ,'fontsize', 30, 'location', 'northeastoutside');
grid on;
