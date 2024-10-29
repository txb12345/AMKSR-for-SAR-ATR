close all;clear all;clc;
tr_filedir = './data sets/SOC/TRAIN_17_DEG/';
tt_filedir = './data sets/SOC/TEST_15_DEG';

[tr_subfiledir,tr_folder] = getSubfolder(tr_filedir);
[tt_subfiledir,tt_folder] = getSubfolder(tt_filedir);
[tr_dat, tr_label, tr_index] = read_dat(tr_subfiledir);
[tt_dat, tt_label, tt_index] = read_dat(tt_subfiledir);

num_tr = size(tr_dat,3);
num_tt = size(tt_dat,3);

idx_class = unique(tr_label);
num_class = length(idx_class);
num_tr_class = zeros(num_class,1);
num_tt_class = zeros(num_class,1);

% choose consecutive views (15)
num_views = 15;
tr_data_SOC = load('./data sets/SOC/tr_data_SOC.mat');
tt_data_SOC = load('./data sets/SOC/tt_data_SOC.mat');
tr_dat_view = tr_data_SOC.tr_data_SOC;
tt_dat_view = tt_data_SOC.tt_data_SOC;
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

A = eye(length(tr_label_view));  
dimension = length(tr_label_view);% the number of the training samples       
num_kernels = 20;
sparsity = 12; % set the sparsity degree
beta_group_SOC = load('beta_group_SOC.mat');
beta_group = beta_group_SOC.beta_group_SOC.beta_group; % load the trained weights
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
%         end
%     end
% end
% save('YTY_kernel_SOC.mat','YTY_kernel');
YTY_kernel_SOC = load('./data sets/SOC/YTY_kernel_SOC.mat');
YTY_kernel = YTY_kernel_SOC.YTY_kernel;
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
    
    [x_test] = KOMP_ONE(0, zTY_test, YTY, A, sparsity);%calculating the sparse codes    
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


      