%  five-fold cross validation, use CNN features with MLP
% Adding weight decay

clearvars;

% Load features and labels of training data
load train/train.mat;

% -- Example: five-fold cross validation, use CNN features
fprintf('Splitting into train/test..\n');

% k-fold cross validation
N = length(train.y);
K=5;
idx = randperm(N);
Nk=floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

%%
Tr = [];
Te = [];
BerNNCNN4_decay2 = [];
for k = 1:K
    % get k'th subgroup in test, others in train
    Te.idxs = idxCV(k,:);
    Tr.idxs = idxCV([1:k-1 k+1:end],:);
    Tr.idxs = Tr.idxs(:);
    Te.y = train.y(Te.idxs);
    Te.X = train.X_cnn(Te.idxs,:);
    Tr.y = train.y(Tr.idxs);
    Tr.X = train.X_cnn(Tr.idxs,:);

    addpath(genpath('/Users/mifei/Dropbox/EPFL/Courses/PCML/PCMLprj/proj2/DeepLearnToolbox-master/'));
      
    setSeed(1);

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.X,2) 500 4]);
    opts.numepochs =  15;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples
    
    % if == 1 => plots trainin error as the NN is trained
    opts.plot               = 1;
    
    nn.learningRate = 2;
    nn.activation_function = 'sigm';
    nn.weightPenaltyL2 = 1e-2;  %  L2 weight decay
    
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
    
    PredNNCNN(:,k)=classVote;
    
    %Compute BER
    BerNNCNN4_decay2 = [BerNNCNN4_decay2,BER(size(nnPred,2), classVote, Te.y)];
    
    fprintf('\nTesting error: %.2f%%\n\n', BerNNCNN4_decay2 * 100 );
    
end

save('result/BerNNCNN4_decay2.mat', 'BerNNCNN4_decay2');

%% For Binary Classification

train.y(find(train.y ~= 4)) = 1;
train.y(find(train.y == 4)) = 2;

Tr = [];
Te = [];
BerBinNNCNN4_decay = [];
for k = 1:K
    % get k'th subgroup in test, others in train
    Te.idxs = idxCV(k,:);
    Tr.idxs = idxCV([1:k-1 k+1:end],:);
    Tr.idxs = Tr.idxs(:);
    Te.y = train.y(Te.idxs);
    Te.X = train.X_cnn(Te.idxs,:);
    Tr.y = train.y(Tr.idxs);
    Tr.X = train.X_cnn(Tr.idxs,:);
     
    setSeed(1);

    % setup NN. The first layer needs to have number of features neurons,
    %  and the last layer the number of classes (here four).
    nn = nnsetup([size(Tr.X,2) 500 2]);
    opts.numepochs =  15;   %  Number of full sweeps through data
    opts.batchsize = 100;  %  Take a mean gradient step over this many samples
    
    % if == 1 => plots trainin error as the NN is trained
    opts.plot               = 1;
    
    nn.learningRate = 2;
    nn.activation_function = 'sigm';
    nn.weightPenaltyL2 = 1e-3;  %  L2 weight decay
    
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
    PredBinNNCNN(:,k)=classVote;  
    % Compute BER
    BerBinNNCNN4_decay = [BerBinNNCNN4_decay,BER(size(nnPred,2), classVote, Te.y)];
    
    fprintf('\nTesting error: %.2f%%\n\n', BerBinNNCNN4_decay * 100 );
    
end

save('result/BerBinNNCNN4_decay.mat', 'BerBinNNCNN4_decay');




%% plot the results for applying MLP for CNN features
load('result/BerNNCNN.mat');
load('result/BerNNCNN2.mat');
load('result/BerNNCNN3.mat');
load('result/BerNNCNN4.mat');
load('result/BerNNCNN5.mat');

%%

p = boxplot([BerNNCNN;BerNNCNN3;BerNNCNN4_decay2;BerNNCNN6; BerNNCNN2; BerNNCNN5]');
ax = gca;
set(gca,'FontSize',24)
title(' MLP using CNN Feature for Multi-class Classification');
ax.XTickLabel = {'One (100)','One (200)','One (500)','One (800)', 'Two (100 10)', 'Two (200 100)','fontsize', 26};
xlabel('Number of Hidden Layers and Size','fontsize', 32);
ylabel('BER','fontsize', 32);
%hLegend = legend(findall(gca,'Tag','Box'), {'2^{-15}','2^{-14}','2^{-13}','2^{-12}','2^{-11}','2^{-10}','2^{-9}','2^{-8}'} ,'fontsize', 30, 'location', 'northeastoutside');
grid on;


%%
load('result/BerBinNNCNN.mat');
load('result/BerBinNNCNN2.mat');
load('result/BerBinNNCNN3.mat');
load('result/BerBinNNCNN4.mat');
load('result/BerBinNNCNN5.mat');


%%
p = boxplot([BerBinNNCNN;BerBinNNCNN3;BerBinNNCNN4_decay; BerBinNNCNN6; BerBinNNCNN2; BerBinNNCNN5]');
ax = gca;
set(gca,'FontSize',24)
title(' MLP using CNN Feature for Binary Classification');
ax.XTickLabel = {'One (100)','One (200)','One (500)', 'One (800)', 'Two (100 10)', 'Two (200 100)','fontsize', 26};
xlabel('Number of Hidden Layers and Size','fontsize', 32);
ylabel('BER','fontsize', 32);
%hLegend = legend(findall(gca,'Tag','Box'), {'2^{-15}','2^{-14}','2^{-13}','2^{-12}','2^{-11}','2^{-10}','2^{-9}','2^{-8}'} ,'fontsize', 30, 'location', 'northeastoutside');
grid on;


