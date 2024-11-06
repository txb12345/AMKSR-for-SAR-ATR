close all;clear all;clc;
tr_dat=load('./data sets/SAMPLE/tr_data_SAMPLE.csv','delimiter',',');
tt_dat=load('./data sets/SAMPLE/tt_data_SAMPLE.csv','delimiter',',');
tr_label = tr_dat(:,1); %the labels of the original training data
tt_label = tt_dat(:,1); %the labels of the original test data
[m1,n1] = size(tr_dat);
tr_dat = tr_dat(1:m1,2:n1)';
[m2,n2] = size(tt_dat);
tt_dat = tt_dat(1:m2,2:n2)';
classNum = 10;  %number of classes
% using PCA to reduce the dimension of data
dat_PCA = [tr_dat,tt_dat];
[dat_coeff,dat_value] = pca(dat_PCA');
tr_dat_value = dat_value(1:m1,:);
tt_dat_value = dat_value((m1+1):end,:);
tr_dat = tr_dat_value(:,1:200);
tt_dat = tt_dat_value(:,1:200);
tr_dat = tr_dat';
tt_dat = tt_dat';

tr_idx_class = unique(tr_label);
tr_num_class = length(tr_idx_class);
tt_idx_class = unique(tt_label);
tt_num_class = length(tt_idx_class);
num_tr_class = zeros(tr_num_class,1);
num_tt_class = zeros(tt_num_class,1);
for i = 1:tr_num_class
    num_tr_class(i) = length(find(tr_label==i));
end
for i = 1:tt_num_class
    num_tt_class(i) = length(find(tt_label==i));
end
num_views = 25;
tr_index_before = 0;
tt_index_before = 0;
for j = 1:tr_num_class
    if j == 1
        for i = 1:num_views
            tr_dat_view{j,i} =  tr_dat(:,i:(num_tr_class(j)-num_views+i));
        end
    end
    if j > 1
        tr_index_before = tr_index_before + num_tr_class(j-1);
        for i = 1:num_views
            tr_dat_view{j,i} =  tr_dat(:,(tr_index_before+i):(tr_index_before + num_tr_class(j)-num_views+i));
        end
    end   
end

for j = 1:tt_num_class
    if j == 1
        for i = 1:num_views
            tt_dat_view{j,i} =  tt_dat(:,i:(num_tt_class(j)-num_views+i));
        end
    end
    if j > 1
        tt_index_before = tt_index_before + num_tt_class(j-1);
        for i = 1:num_views
            tt_dat_view{j,i} =  tt_dat(:,(tt_index_before+i):(tt_index_before + num_tt_class(j)-num_views+i));
        end
    end   
end

for j = 1:num_views
    tr_dat_view_total{j} = [];
    tt_dat_view_total{j} = [];
    for i = 1:tr_num_class
        tr_dat_view_total{j} = cat(2, tr_dat_view_total{j}, tr_dat_view{i,j});
    end
    for i = 1:tt_num_class
        tt_dat_view_total{j} = cat(2, tt_dat_view_total{j}, tt_dat_view{i,j});
    end
end

%Importing the training and testing data
for j = 1:num_views
    tr_dat_view_new{j} = [];
    tt_dat_view_new{j} = [];
    for i = 1:size(tr_dat_view_total{j},2)
        tr_dat_view_new{j} = [tr_dat_view_new{j},tr_dat_view_total{j}(:,i)];
    end
    for i = 1:size(tt_dat_view_total{j},2)
        tt_dat_view_new{j} = [tt_dat_view_new{j},tt_dat_view_total{j}(:,i)];
    end
    % % data_reduce
    tr_dat_original = (tr_dat_view_new{j})';
    tt_dat_original = (tt_dat_view_new{j})';
    tr_dat_ACD_pca_view{j} = tr_dat_original(:,1:100); %acquire the training data features
    tt_dat_ACD_pca_view{j} = tt_dat_original(:,1:100); %acquire the test data features
end

idx_class = unique(tr_label);
num_class = length(idx_class);
num_tr_class = zeros(num_class,1);
num_tt_class = zeros(num_class,1);

tr_dat_view = tr_dat_ACD_pca_view;
tt_dat_view = tt_dat_ACD_pca_view;
for i = 1:num_views
    tr_dat_view{i} = zscore((tr_dat_view{i}(:,1:25))');
    tt_dat_view{i} = zscore((tt_dat_view{i}(:,1:25))');
end

tr_label_view = [];tt_label_view = [];
for i = 1:num_class
    num_tr_class(i) = length(find(tr_label==i));
    num_tt_class(i) = length(find(tt_label==i));
    for j = 1:(num_tr_class(i)-num_views+1)
        tr_label_view  = [tr_label_view;i]; %the labels of the multiview training data
    end
    for k = 1:(num_tt_class(i)-num_views+1)
        tt_label_view = [tt_label_view;i]; %the labels of the multiview test data
    end 
end

% param.K=length(tr_label_view);
% param.iter=100;
% param.lambda        = 0.20; % not more than 20 non-zeros coefficients
% param.mode          = 2;   % penalized formulation
% for m = 1:num_views
%     tr_dat_view_D{m} = mexTrainDL(tr_dat_view{m}, param);
% end
tr_dat_view_D = load('tr_dat_view_D.mat'); 
tr_dat_view_D = tr_dat_view_D.tr_dat_view_D;

A = eye(length(tr_label_view));  
dimension = length(tr_label_view);
%ideal kernel 
Q = zeros(dimension, dimension);            
beta_group_error = [];
num_kernels = 20;
sparsity = 12;
% training the kernel matrices of different kernels
for p = 1:num_kernels
    YTY_kernel{p} = zeros(dimension,dimension);
    for k1 = 1:dimension
        for k2 = 1:dimension
            for i = 1:num_views
                YTY_view{i}(k1, k2) = kernel_function(tr_dat_view{i}(:, k1),tr_dat_view{i}(:, k2),p);  %calculate the kernel matrices of different views
                YTY_view{i}(k2, k1) = YTY_view{i}(k1, k2);
                YTY_kernel{p}(k1, k2) = YTY_kernel{p}(k1, k2) + YTY_view{i}(k1, k2); 
            end
            YTY_kernel{p}(k1, k2) = YTY_kernel{p}(k1, k2)/num_views;
            YTY_kernel{p}(k2, k1) = YTY_kernel{p}(k1, k2);

        end
    end
end
for m = 1:num_kernels
    residual{m} = zeros(length(tr_label_view),num_class);
end
% training phase
beta_group = zeros(num_kernels,1);
beta_group(15) = 1;
for t = 1 : 5
    YTY = zeros(dimension, dimension);       
    for m = 1:num_kernels
        YTY = YTY + beta_group(m) * YTY_kernel{m};
    end  
    for j = 1:dimension         
        zTY = zeros(1, dimension);
        zTz_train = 0;
        for p = 1:num_kernels
            zTY_kernel{p} = zeros(1, dimension);
            zTz_{p} = 0;
            for m = 1:num_views
                zTY_view{m} = zeros(1, dimension);
                for k = 1:dimension
                    zTY_view{m}(1, k) = kernel_function(tr_dat_view_D{m}(:, j),tr_dat_view{m}(:, k),p);           
                end
                zTY_kernel{p} = zTY_kernel{p} + zTY_view{m};
                zTz_{p} = zTz_{p} + kernel_function(tr_dat_view_D{m}(:, j),tr_dat_view_D{m}(:, j),p); 
            end 
            zTY_kernel{p} = zTY_kernel{p}/num_views;
            zTz_{p} = zTz_{p}/num_views;
            zTY = zTY + beta_group(p) * zTY_kernel{p};
            zTz_train = zTz_train + beta_group(p) * zTz_{p};
            
        end
        x_composite = KOMP_ONE(0, zTY, YTY, A, sparsity); % use the KOMP to optimize algorithm
        for i = 1:num_class
            code_composite = x_composite(tr_label_view == i); %code is the sparse code
            residual_composite(j, i) = zTz_train - 2 * zTY(tr_label_view == i) * code_composite + code_composite' * YTY(tr_label_view == i, tr_label_view == i) * code_composite; %calculate the reconstruction error of the composite kernel
        end
        for i = 1:num_class
            for m = 1:num_kernels
                code{m} = x_composite(tr_label_view == i); %code is the sparse code
                residual{m}(j, i) = zTz_{m} - 2 * zTY_kernel{m}(tr_label_view == i) * code{m} + code{m}' *  YTY_kernel{m}(tr_label_view == i, tr_label_view == i) * code{m}; %calculate the reconstruction error of each kernel
            end
        end
        
    end    
    for m = 1:num_kernels
        [value{m}, index{m}] = min(residual{m} , [], 2); % acquire the minimal reconstruction error
    end
    [value_composite, index_composite] = min(residual_composite, [], 2);
    current_kernel = [];current_kernel_index = [];
    correct_composite = 0;
    for i = 1:length(tr_label_view)
        if index_composite(i) == tr_label_view(i)
            correct_composite = correct_composite + 1;
            current_kernel = [current_kernel;1];
        else
            current_kernel = [current_kernel;0];
            current_kernel_index = [current_kernel_index;i];
        end

    end
    for m = 1:num_kernels
        correct_count{m} = 0;
        choose_kernel{m} = [];
        for i = 1:length(tr_label_view)
            if index{m}(i) == tr_label_view(i)
                correct_count{m} = correct_count{m} + 1;
                choose_kernel{m} = [choose_kernel{m},1];
            else
                choose_kernel{m} = [choose_kernel{m},0];
            end

        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %ideal kernel
    for i = 1:length(tr_label_view)
        for j = 1:length(tr_label_view)
            if tr_label_view(j) == tr_label_view(i)
                Q(j, i) = 1;
            else 
                Q(j, i) = 0;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %kernel alignment, measure differences.
    alignment = zeros(num_kernels,1);
    for m = 1:num_kernels
        alignment(m) = sum(sum(YTY_kernel{m} * Q))/(dimension * sqrt(sum(sum(YTY_kernel{m} * YTY_kernel{m}))));
    end
    
    [alignment_sort_value,alignment_sort_index] = sort(alignment,'descend');
    %select new kernel
    miss_current_kernel_composite = sum(1 - current_kernel);
    miss_choose_kernel_group = [];
    for m = 1:num_kernels
        kernel_accuracy = sum(choose_kernel{m}(current_kernel_index))/miss_current_kernel_composite;
        miss_choose_kernel_group = [miss_choose_kernel_group, kernel_accuracy];
    end
    %%%%%%%%%%%%%%%%%%%% Algorithm 1 in the paper %%%%%%%%%%%%%%%%%%%%%%%%%%
    miss_choose_kernel_value = [];
    for i = 1:num_kernels
        miss_choose_kernel_value(i) = miss_choose_kernel_group(alignment_sort_index(i));%sort according to the recognition scores
    end
    mu = 0.02;
    [miss_choose_kernel_value_max,miss_choose_kernel_index_max] = max(miss_choose_kernel_value);%find the index of the maximal recognition score
    for i = 1:miss_choose_kernel_index_max
        if miss_choose_kernel_value(i) + mu > miss_choose_kernel_value_max 
            miss_choose_kernel_index_selected = i;
            break;
        end
    end 
    choose_kernel_index = alignment_sort_index(miss_choose_kernel_index_max);
    disp(choose_kernel_index)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Wnewkernel = sum(double(choose_kernel{choose_kernel_index}&(1-current_kernel)))/sum(double((1-choose_kernel{choose_kernel_index})|(1-current_kernel)));
    Wcurrentkernel = sum(double((1-choose_kernel{choose_kernel_index})&current_kernel))/sum(double((1-choose_kernel{choose_kernel_index})|(1-current_kernel))); %calculate the weights of dfferent kernels
  
    %update weights
    beta_group_sum = 0;
    beta_group_previous = beta_group;
    for i = 1:num_kernels
        if i ~= choose_kernel_index
            beta_group(i) = beta_group(i) * Wcurrentkernel;                           
        else
            beta_group(i) = Wnewkernel;
        end
        beta_group_sum = beta_group_sum + beta_group(i);       
    end 
    for i = 1:num_kernels
        beta_group(i) = beta_group(i)/beta_group_sum;         
    end   
    %calculate error
    beta_group_error(t) = sqrt(sum((beta_group_previous - beta_group).^2));  % determine whether convergence has occurred according to weight errors
    
end
% test phase
YTY = zeros(dimension, dimension);  
for p = 1:num_kernels
    if beta_group(p) == 0
        continue;
    end
    YTY = YTY + beta_group(p) * YTY_kernel{p};
end
for j = 1:length(tt_label_view)
    tic
    zTY_test = zeros(1, dimension);
    zTz_test = 0;
    for p = 1:num_kernels
        if beta_group(p) == 0
            continue;
        end
        zTY_kernel_test{p} = zeros(1, dimension);
        zTz{p} = 0;
        for m = 1:num_views
            zTY_view_test{m} = zeros(1, dimension);
            for k = 1:dimension
                zTY_view_test{m}(1, k) = kernel_function(tt_dat_view{m}(:, j),tr_dat_view{m}(:, k),p);           
            end
            zTY_kernel_test{p} = zTY_kernel_test{p} + zTY_view_test{m};
            zTz{p} = zTz{p} + kernel_function(tt_dat_view{m}(:, j),tt_dat_view{m}(:, j),p); 
        end

        zTY_kernel_test{p} = zTY_kernel_test{p}/num_views;
        zTz{p} = zTz{p}/num_views;
        zTY_test = zTY_test + beta_group(p) * zTY_kernel_test{p};
        zTz_test = zTz_test + beta_group(p) * zTz{p};
    end
    [x_test] = KOMP_ONE(0, zTY_test, YTY, A, sparsity);
    %Calculation of K(z, z)
    for i = 1:num_class       
        code_test = x_test(tr_label_view == i); %code is the sparse code
        residual_test(j, i) = zTz_test - 2 * zTY_test(tr_label_view == i) * code_test + code_test' * YTY(tr_label_view == i, tr_label_view == i) * code_test; %calculate the reconstruction error
      
    end
    toc  
end

[value_test, index_test] = min(residual_test, [], 2);
c1 = tt_label_view;
c2 = index_test;
c3 = residual_test;
c3 = 1./(c3+0.000001);

for i = 1:size(c3,1)
    c3(i,:) = func_softmax(c3(i,:));
end

true_labels = c1;
predicted_labels = c2;
scores = c3;
conf_mat = confusionmat(true_labels,predicted_labels);
Precision = 0;
Recall = 0;
for i = 1:num_class
    tp = conf_mat(i,i);
    fp = sum(conf_mat(:,i)) - tp;
    fn = sum(conf_mat(i,:)) - tp;
    precision(i) = tp/(tp + fp);
    recall(i) = tp/(tp + fn);
    Precision = Precision + precision(i);
    Recall = Recall + recall(i);
end
Accuracy = length(find(c1==c2))/size(c1,1)
Precision = Precision/num_class
Recall = Recall/num_class

function y = func_softmax(x)
    x = x - max(x);
    y = exp(x) ./ sum(exp(x));
end

