close all;clear all;clc;
%%%%%%%%%%%% Prepare multiview data and corresponding labels %%%%%%%%%%
data_master=load('./data sets/SAR-ACD/data_SAR_ACD.csv','delimiter',',');
[data_r,data_c] = size(data_master);
indices = cell2mat(struct2cell(load('indices.mat')));
test = (indices == 1);
train = ~test;
tr_dat = data_master(train,1:data_c);
tt_dat = data_master(test,1:data_c); 
tr_label = tr_dat(:,1); 
tt_label = tt_dat(:,1); 
[m1,n1] = size(tr_dat); 
tr_dat = tr_dat(1:m1,2:n1)';
[m2,n2] = size(tt_dat);
tt_dat = tt_dat(1:m2,2:n2)';
classNum = 6;       
%PCA
dat_PCA = [tr_dat,tt_dat];
[dat_coeff,dat_value] = pca(dat_PCA');
tr_dat_value = dat_value(1:m1,:);
tt_dat_value = dat_value((m1+1):data_r,:);
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
% %choose consecutive views (9)
num_views = 8;
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
    tr_dat_ACD_pca_view{j} = tr_dat_original(:,1:100);
    tt_dat_ACD_pca_view{j} = tt_dat_original(:,1:100);
end

idx_class = unique(tr_label);
num_class = length(idx_class);
num_tr_class = zeros(num_class,1);
num_tt_class = zeros(num_class,1);

tr_dat_view = tr_dat_ACD_pca_view;
tt_dat_view = tt_dat_ACD_pca_view;
for i = 1:num_views
    tr_dat_view{i} = zscore((tr_dat_view{i}(:,1:10))');
    tt_dat_view{i} = zscore((tt_dat_view{i}(:,1:10))');
end

tr_label_view = [];tt_label_view = [];
for i = 1:num_class
    num_tr_class(i) = length(find(tr_label==i));
    num_tt_class(i) = length(find(tt_label==i));
    for j = 1:(num_tr_class(i)-num_views+1)
        tr_label_view  = [tr_label_view;i];
    end
    for k = 1:(num_tt_class(i)-num_views+1)
        tt_label_view = [tt_label_view;i];
    end 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = eye(length(tr_label_view));  
dimension = length(tr_label_view); % the number of the training samples    
num_kernels = 20;
sparsity = 12; % set the sparsity degree
beta_group_SAR_ACD = load('beta_group_SAR_ACD.mat');
beta_group = beta_group_SAR_ACD.beta_group; % load the trained weights
% for p = 1:num_kernels
%     YTY_kernel{p} = zeros(dimension,dimension);
%     for k1 = 1:dimension
%         for k2 = 1:dimension  
%             for i = 1:num_views
%                 YTY_view{i}(k1, k2) = kernel_function(tr_dat_view{i}(:, k1),tr_dat_view{i}(:, k2),p);
%                 YTY_view{i}(k2, k1) = YTY_view{i}(k1, k2);
%                 YTY_kernel{p}(k1, k2) = YTY_kernel{p}(k1, k2) + YTY_view{i}(k1, k2); 
%             end
%             YTY_kernel{p}(k1, k2) = YTY_kernel{p}(k1, k2)/num_views;
%             YTY_kernel{p}(k2, k1) = YTY_kernel{p}(k1, k2);
% 
%         end
%     end
% end
% save('YTY_kernel_ACD.mat','YTY_kernel');
YTY_kernel_ACD = load('./data sets/SAR-ACD/YTY_kernel_ACD.mat');
YTY_kernel = YTY_kernel_ACD.YTY_kernel;
YTY = zeros(dimension, dimension);  
for p = 1:num_kernels
    if beta_group(p) == 0 % eliminate the kernels with weight of zero
        continue;
    end
    YTY = YTY + beta_group(p) * YTY_kernel{p};
end
for j = 1:length(tt_label_view)
    tic
    zTY_test = zeros(1, dimension);
    zTz_test = 0;
    for p = 1:num_kernels
        if beta_group(p) == 0 % eliminate the kernels with weight of zero
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
        code_test = x_test(tr_label_view == i); %code is a coefficient vector 
        residual_test(j, i) = zTz_test - 2 * zTY_test(tr_label_view == i) * code_test + code_test' * YTY(tr_label_view == i, tr_label_view == i) * code_test;
      
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

numclasses = num_class;
true_labels = c1;
predicted_labels = c2;
scores = c3;
conf_mat = confusionmat(true_labels,predicted_labels);
Precision = 0;
Recall = 0;
for i = 1:numclasses
    tp = conf_mat(i,i);
    fp = sum(conf_mat(:,i)) - tp;
    fn = sum(conf_mat(i,:)) - tp;
    precision(i) = tp/(tp + fp);
    recall(i) = tp/(tp + fn);
    Precision = Precision + precision(i);
    Recall = Recall + recall(i);
end
Accuracy = length(find(c1==c2))/size(c1,1)
Precision = Precision/numclasses
Recall = Recall/numclasses


%functions
function y = func_softmax(x)
    x = x - max(x);
    y = exp(x) ./ sum(exp(x));
end


function [dat, lab, index] = read_dat(src)
    lab = [];
    index = 0;
    dat = [];
    new_width = 50;
    new_height = 50;
    for i = 1:length(src)  
       %src(i)
       img_path_list = dir(src(i));
       img_num = length(img_path_list);%获取图像总数量 
       for j = 3:img_num %逐一读取图像
            image_name = img_path_list(j).name;% 图像名
            %fprintf('%s\n',image_name);
            %if strcmp(image_name(end-3:end), 'jpeg')
            image_path = strcat(src(i),'\', image_name);
	        image =  imread(image_path);
            [width,height] = size(image);
            lab = [lab;i];%创建标签
            index = index + 1;
            left = floor((width - new_width) / 2);%取整
            top = floor((height - new_height) / 2);
            right = floor((width + new_width) / 2);
            bottom = floor((height + new_height) / 2);
            image = image(left:(right-1),top:(bottom-1));
            dat = cat(3,dat,image);
            %end
    
       end
    
    end
end
% 
function [s, folder] = getSubfolder(src)
    folder = dir(src);
    subfolder = folder(3:end);
    s = [];
    for i = 1:length(subfolder)
        path = [src, '\', subfolder(i).name];
        if ~isfile(path)
            s = [s, string(path)];
        end
        if isfolder(path)
            subpath = getSubfolder(path);
            s = [s, subpath];
        end
    end
end


      