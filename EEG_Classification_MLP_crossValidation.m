load SubHandData

train_feature = [];
for i=1:180  %sample
    for j=1:30  %canal
        bp_total = bandpower(bandpower(double(TrainX(j,:,i))));
        train_feature(1,j,i) = max(abs(TrainX(j,:,i))); %max
        train_feature(2,j,i) = var(double(TrainX(j,:,i))); %var
        train_feature(3,j,i) = meanfreq(double(TrainX(j,:,i)));
        train_feature(4,j,i) = medfreq(double(TrainX(j,:,i)));
        train_feature(5,j,i) = bandpower(double(TrainX(j,:,i)),256,[2,8])/bp_total;
        train_feature(6,j,i) = bandpower(double(TrainX(j,:,i)),256,[9,15])/bp_total;
        train_feature(7,j,i) = bandpower(double(TrainX(j,:,i)),256,[16,22])/bp_total;
        train_feature(8,j,i) = bandpower(double(TrainX(j,:,i)),256,[23,29])/bp_total;
        train_feature(9,j,i) = bandpower(double(TrainX(j,:,i)),256,[30,36])/bp_total;
        train_feature(10,j,i) = bandpower(double(TrainX(j,:,i)),256,[37,43])/bp_total;
        train_feature(11,j,i) = bandpower(double(TrainX(j,:,i)),256,[44,50])/bp_total;
    end
end

%Normalization
[Normalized_Train_Feature1,xPS] = mapminmax(train_feature(1,:,:));
[Normalized_Train_Feature2,xPS] = mapminmax(train_feature(2,:,:));
[Normalized_Train_Feature3,xPS] = mapminmax(train_feature(3,:,:));
[Normalized_Train_Feature4,xPS] = mapminmax(train_feature(4,:,:));
[Normalized_Train_Feature5,xPS] = mapminmax(train_feature(5,:,:));
[Normalized_Train_Feature6,xPS] = mapminmax(train_feature(6,:,:));
[Normalized_Train_Feature7,xPS] = mapminmax(train_feature(7,:,:));
[Normalized_Train_Feature8,xPS] = mapminmax(train_feature(8,:,:));
[Normalized_Train_Feature9,xPS] = mapminmax(train_feature(9,:,:));
[Normalized_Train_Feature10,xPS] = mapminmax(train_feature(10,:,:));
[Normalized_Train_Feature11,xPS] = mapminmax(train_feature(11,:,:));


%select useful features using J 
H_indices = find(Trainy==1) ;
M_indices = find(Trainy==0) ;
J = zeros(11,30);
    for j=1:30
        u1 = mean(Normalized_Train_Feature1(1,j,H_indices));
        s1 = var(Normalized_Train_Feature1(1,j,H_indices));
        u2 = mean(Normalized_Train_Feature1(1,j,M_indices));
        s2 = var(Normalized_Train_Feature1(1,j,H_indices));
        u0 = mean(Normalized_Train_Feature1(1,j,:));
        Sw = s1 + s2;
        Sb = (u1-u0)^2 + (u2-u0)^2;
        J(1,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature2(1,j,H_indices));
    s1 = var(Normalized_Train_Feature2(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature2(1,j,M_indices));
    s2 = var(Normalized_Train_Feature2(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature2(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(2,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature3(1,j,H_indices));
    s1 = var(Normalized_Train_Feature3(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature3(1,j,M_indices));
    s2 = var(Normalized_Train_Feature3(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature3(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(3,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature4(1,j,H_indices));
    s1 = var(Normalized_Train_Feature4(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature4(1,j,M_indices));
    s2 = var(Normalized_Train_Feature4(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature4(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(4,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature5(1,j,H_indices));
    s1 = var(Normalized_Train_Feature5(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature5(1,j,M_indices));
    s2 = var(Normalized_Train_Feature5(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature5(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(5,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature6(1,j,H_indices));
    s1 = var(Normalized_Train_Feature6(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature6(1,j,M_indices));
    s2 = var(Normalized_Train_Feature6(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature6(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(6,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature7(1,j,H_indices));
    s1 = var(Normalized_Train_Feature7(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature7(1,j,M_indices));
    s2 = var(Normalized_Train_Feature7(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature7(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(7,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature8(1,j,H_indices));
    s1 = var(Normalized_Train_Feature8(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature8(1,j,M_indices));
    s2 = var(Normalized_Train_Feature8(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature8(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(8,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature9(1,j,H_indices));
    s1 = var(Normalized_Train_Feature9(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature9(1,j,M_indices));
    s2 = var(Normalized_Train_Feature9(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature9(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(9,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature10(1,j,H_indices));
    s1 = var(Normalized_Train_Feature10(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature10(1,j,M_indices));
    s2 = var(Normalized_Train_Feature10(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature10(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(10,j) = Sb/Sw ;
    end
    for j=1:30
    u1 = mean(Normalized_Train_Feature11(1,j,H_indices));
    s1 = var(Normalized_Train_Feature11(1,j,H_indices));
    u2 = mean(Normalized_Train_Feature11(1,j,M_indices));
    s2 = var(Normalized_Train_Feature11(1,j,H_indices));
    u0 = mean(Normalized_Train_Feature11(1,j,:));
    Sw = s1 + s2;
    Sb = (u1-u0)^2 + (u2-u0)^2;
    J(11,j) = Sb/Sw ;
    end
    %%%mean of J for each feature for all channels
    J_mean = mean(J,2);
    [J_sorted,I_sorted] = sort(J,2,'descend');
    %%indicates that powerband features are more important
%     figureplot
%     plot3(Normalized_Train_Feature7(1,2,H_indices),Normalized_Train_Feature7(1,25,H_indices),Normalized_Train_Feature7(1,86,H_indices),'*r') ;
%     hold on
%     plot3(Normalized_Train_Feature9(1,10,M_indices),Normalized_Train_Feature9(1,2,M_indices),Normalized_Train_Feature9(1,54,M_indices),'og') ;
%     title('Fetures #2, #3, #4') ;


%%MLP network
% 
% %%feature selection---data was NORMALIZED before
Train_Features = zeros(99,180);
for i =1:180
   for j=1:9
       Train_Features(j,i) = Normalized_Train_Feature1(1,I_sorted(1,j),i);
       Train_Features(j+9,i) = Normalized_Train_Feature2(1,I_sorted(2,j),i);
       Train_Features(j+18,i) = Normalized_Train_Feature3(1,I_sorted(3,j),i);
       Train_Features(j+27,i) = Normalized_Train_Feature4(1,I_sorted(4,j),i);
       Train_Features(j+36,i) = Normalized_Train_Feature5(1,I_sorted(5,j),i);
       Train_Features(j+45,i) = Normalized_Train_Feature6(1,I_sorted(6,j),i);
       Train_Features(j+54,i) = Normalized_Train_Feature7(1,I_sorted(7,j),i);
       Train_Features(j+63,i) = Normalized_Train_Feature8(1,I_sorted(8,j),i);
       Train_Features(j+72,i) = Normalized_Train_Feature9(1,I_sorted(9,j),i);
       Train_Features(j+81,i) = Normalized_Train_Feature10(1,I_sorted(10,j),i);
       Train_Features(j+90,i) = Normalized_Train_Feature11(1,I_sorted(11,j),i);
   end
end

X_Features = Train_Features([1,2,10,11,19,20,28,29,30,37,38,39,46,47,48,55,56,57,64,65,66,73,74,75,81,82,83,90,91,92,49,50,58,59],:) ;
Train_Label = Trainy ;
% 
% for N=1:50
%     ACC = 0 ;
%     for k=1:5
%         train_indices = [1:(k-1)*36,k*36+1:180]; 
%         valid_indices = (k-1)*36+1:k*36 ;
%         
%         X_Features2 = X_Features(:,train_indices) ;
%         ValX = X_Features(:,valid_indices) ;
%         Train_Label2 = Trainy(:,train_indices) ;
%         ValY = Trainy(:,valid_indices) ;
%         
%         net = patternnet(N);
%         net = train(net,X_Features,Train_Label);
% 
%         predict_y = net(ValX);
% 
%         [maxval,mindx] = max(predict_y) ;
%         p_ValY = zeros(2,36) ;
%         p_ValY(1,find(mindx==1)) = 1 ;
%         p_ValY(2,find(mindx==2)) = 1 ;
% 
%         ACC = ACC + length(find(p_ValY(1,:)==ValY(1,:))) ;
%     end 
% 
%     ACCMat(N) = ACC/180 ;
% end
% % %%conclude that best N=31
net = patternnet(31);
net = train(net,X_Features,Train_Label);

predict_ymlp = net(X_Features);

[maxval,mindx] = max(predict_ymlp) ;
p_ValY = zeros(2,180) ;
p_ValY(1,find(mindx==1)) = 1 ;
p_ValY(2,find(mindx==2)) = 1 ;

ACC =  length(find(p_ValY(1,:)==Train_Label(1,:))) / 180

%%Finding Test_Data labels

test_feature = zeros(11,30,60);
for i=1:60  %sample
    for j=1:30  %canal
        bp_total = bandpower(bandpower(double(TestX(j,:,i))));
        test_feature(1,j,i) = max(abs(TestX(j,:,i))); %max
        test_feature(2,j,i) = var(double(TestX(j,:,i))); %var
        test_feature(3,j,i) = meanfreq(double(TestX(j,:,i)));
        test_feature(4,j,i) = medfreq(double(TestX(j,:,i)));
        test_feature(5,j,i) = bandpower(double(TestX(j,:,i)),256,[2,8])/bp_total;
        test_feature(6,j,i) = bandpower(double(TestX(j,:,i)),256,[9,15])/bp_total;
        test_feature(7,j,i) = bandpower(double(TestX(j,:,i)),256,[16,22])/bp_total;
        test_feature(8,j,i) = bandpower(double(TestX(j,:,i)),256,[23,29])/bp_total;
        test_feature(9,j,i) = bandpower(double(TestX(j,:,i)),256,[30,36])/bp_total;
        test_feature(10,j,i) = bandpower(double(TestX(j,:,i)),256,[37,43])/bp_total;
        test_feature(11,j,i) = bandpower(double(TestX(j,:,i)),256,[44,50])/bp_total;
    end
end

[Normalized_Test_Feature1,xPS] = mapminmax(test_feature(1,:,:));
[Normalized_Test_Feature2,xPS] = mapminmax(test_feature(2,:,:));
[Normalized_Test_Feature3,xPS] = mapminmax(test_feature(3,:,:));
[Normalized_Test_Feature4,xPS] = mapminmax(test_feature(4,:,:));
[Normalized_Test_Feature5,xPS] = mapminmax(test_feature(5,:,:));
[Normalized_Test_Feature6,xPS] = mapminmax(test_feature(6,:,:));
[Normalized_Test_Feature7,xPS] = mapminmax(test_feature(7,:,:));
[Normalized_Test_Feature8,xPS] = mapminmax(test_feature(8,:,:));
[Normalized_Test_Feature9,xPS] = mapminmax(test_feature(9,:,:));
[Normalized_Test_Feature10,xPS] = mapminmax(test_feature(10,:,:));
[Normalized_Test_Feature11,xPS] = mapminmax(test_feature(11,:,:));


Test_Features = zeros(99,60);
for i =1:60
   for j=1:9
       Test_Features(j,i) = Normalized_Test_Feature1(1,I_sorted(1,j),i);
       Test_Features(j+9,i) = Normalized_Test_Feature2(1,I_sorted(2,j),i);
       Test_Features(j+18,i) = Normalized_Test_Feature3(1,I_sorted(3,j),i);
       Test_Features(j+27,i) = Normalized_Test_Feature4(1,I_sorted(4,j),i);
       Test_Features(j+36,i) = Normalized_Test_Feature5(1,I_sorted(5,j),i);
       Test_Features(j+45,i) = Normalized_Test_Feature6(1,I_sorted(6,j),i);
       Test_Features(j+54,i) = Normalized_Test_Feature7(1,I_sorted(7,j),i);
       Test_Features(j+63,i) = Normalized_Test_Feature8(1,I_sorted(8,j),i);
       Test_Features(j+72,i) = Normalized_Test_Feature9(1,I_sorted(9,j),i);
       Test_Features(j+81,i) = Normalized_Test_Feature10(1,I_sorted(10,j),i);
       Test_Features(j+90,i) = Normalized_Test_Feature11(1,I_sorted(11,j),i);
   end
end
X2_Features = Test_Features([1,2,10,11,19,20,28,29,30,37,38,39,46,47,48,55,56,57,64,65,66,73,74,75,81,82,83,90,91,92,49,50,58,59],:) ;

predict_ymlpt = net(X2_Features);

[maxval,mindx] = max(predict_ymlpt) ;
p_TestY0 = zeros(2,60) ;
p_TestY0(1,find(mindx==1)) = 1 ;
p_TestY0(2,find(mindx==2)) = 1 ;


%RBF network


% Train
%%Train_Features must be imported
X3_Features = Train_Features([1,2,10,11,19,20,28,29,30,37,38,39,46,47,48,55,56,57,64,65,66,73,74,75,81,82,83,90,91,92,49,50,58,59],:) ;
Train_Label = Trainy ;

spreadMat = [.1,.5,.9,1.5,2] ;
NMat = [5,10,15,20,25] ;

%%5-fold cross validation
% for s = 1:5
%     spread = spreadMat(s) ;
%     for n = 1:5 
%         Maxnumber = NMat(n) ;
%         ACC = 0 ;
%         % 5-fold cross-validation
%         for k=1:5
%             train_indices = [1:(k-1)*36,k*36+1:180]; 
%             valid_indices = (k-1)*36+1:k*36 ;
% 
%             X3_Features2 = X3_Features(:,train_indices) ;
%             ValX = X3_Features(:,valid_indices) ;
%             Train_Label2 = Trainy(:,train_indices) ;
%             ValY = Trainy(:,valid_indices) ;
% 
%             net = newrb(X3_Features,Train_Label,10^-5,spread,Maxnumber) ;
%             predict_y = net(ValX);
%             
%             [maxval,mindx] = max(predict_y) ;
%             p_ValY = zeros(2,36) ;
%             p_ValY(1,find(mindx==1)) = 1 ;
%             p_ValY(2,find(mindx==2)) = 1 ;
% 
%             ACC = ACC + length(find(p_ValY(1,:)==ValY(1,:))) ;
%         end
%         ACCMat(s,n) = ACC/180 ;
%     end
% end
% s= .5 & N = 25
% % Classification
spread = spreadMat(2) ; % Best parameter found in training step
Maxnumber = NMat(5) ; % Best parameter found in training step


net = newrb(X3_Features,Train_Label,10^-5,spread,Maxnumber) ;

predict_yr = net(X3_Features);

[maxval,mindx] = max(predict_yr) ;
p_ValY = zeros(2,180) ;
p_ValY(1,find(mindx==1)) = 1 ;
p_ValY(2,find(mindx==2)) = 1 ;

ACC =  length(find(p_ValY(1,:)==Train_Label(1,:))) / 180
%%finding Test Data Labels
%%we have Test_Features from t0
Xtest_Features = Test_Features([1,2,10,11,19,20,28,29,30,37,38,39,46,47,48,55,56,57,64,65,66,73,74,75,81,82,83,90,91,92,49,50,58,59],:) ;

predict_yrt = net(Xtest_Features);

[maxval,mindx] = max(predict_yrt) ;
p_TestY1 = zeros(2,60) ;
p_TestY1(1,find(mindx==1)) = 1 ;
p_TestY1(2,find(mindx==2)) = 1 ;
