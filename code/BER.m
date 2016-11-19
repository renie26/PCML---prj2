% compute the balanced error rate

function result = BER(C, classVote, truth)
%truth = Te.y;
correct_pred = [];
for i = 1:C
   index = truth==i; % compute index of true labels = i
   %disp((classVote(index)==i))
   correct_pred = [correct_pred,sum((classVote(index)~=i))/sum(index)];
end

result = mean(correct_pred);




