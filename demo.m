close all;clear all;clc;
tr_filedir = 'C:\Users\Administrator\Desktop\SOC\TRAIN_17_DEG\';
tt_filedir = 'C:\Users\Administrator\Desktop\SOC\TEST_15_DEG\';

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

% %choose consecutive views (10)
num_views = 13;
tr_dat_pca_view_v6_13views = load('tr_dat_pca_view_v6_13views.mat');
tt_dat_pca_view_v6_13views = load('tt_dat_pca_view_v6_13views.mat');
tr_dat_view = tr_dat_pca_view_v6_13views.tr_dat_pca_view_v6_13views;
tt_dat_view = tt_dat_pca_view_v6_13views.tt_dat_pca_view_v6_13views;
for i = 1:num_views
    tr_dat_view{i} = zscore((tr_dat_view{i}(:,1:10))');
    tt_dat_view{i} = zscore((tt_dat_view{i}(:,1:10))');
end
tr_label_view_supervise = zeros(num_class,size(tr_dat_view{1},2)); 
tt_label_view_supervise = zeros(num_class,size(tt_dat_view{1},2));
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

tr_dat_view_D = load('tr_dat_view_D.mat');
tr_dat_view_D = tr_dat_view_D.tr_dat_view_D;

A = eye(length(tr_label_view));  
dimension = length(tr_label_view);
%ideal kernel 
Q = zeros(dimension, dimension);            
beta_group_error = [];
num_kernels = 20;
sparsity = 12;
for m = 1:num_kernels
    residual{m} = zeros(length(tr_label_view),num_class);
end
% 权重参数
beta_group = zeros(num_kernels,1);
beta_group(12) = 1; %
for t = 1:10
    for p = 1:num_kernels
        YTY_kernel{p} = zeros(dimension,dimension);
        for k1 = 1:dimension
            for k2 = 1:dimension 
                for i = 1:num_views
                    YTY_view{i}(k1, k2) = kernel_function(tr_dat_view{i}(:, k1),tr_dat_view{i}(:, k2),p);
                    YTY_view{i}(k2, k1) = YTY_view{i}(k1, k2);
                    YTY_kernel{p}(k1, k2) = YTY_kernel{p}(k1, k2) + YTY_view{i}(k1, k2); 
                end
                YTY_kernel{p}(k1, k2) = YTY_kernel{p}(k1, k2)/num_views;
                YTY_kernel{p}(k2, k1) = YTY_kernel{p}(k1, k2);
            end
        end
    end

    YTY = zeros(dimension, dimension);       
    for m = 1:num_kernels
        YTY = YTY + beta_group(m) * YTY_kernel{m};
    end
    %训练阶段   
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
        x_composite = KOMP_ONE(0, zTY, YTY, A, dimension, sparsity); 
        %Calculation of K(z, z)
        %zTz = 1;
        for i = 1:num_class
            code_composite = x_composite(tr_label_view == i); %code is a coefficient vector 
            residual_composite(j, i) = zTz_train - 2 * zTY(tr_label_view == i) * code_composite + code_composite' * YTY(tr_label_view == i, tr_label_view == i) * code_composite;
        end
        %Calculation of K(z, z)
        for i = 1:num_class
            for m = 1:num_kernels
                code{m} = x_composite(tr_label_view == i); %code is a coefficient vector 
                residual{m}(j, i) = zTz_{m} - 2 * zTY_kernel{m}(tr_label_view == i) * code{m} + code{m}' *  YTY_kernel{m}(tr_label_view == i, tr_label_view == i) * code{m};
            end
        end
        

    end
    
    %Classification, finding the corresponding class with minimum residual
    for m = 1:num_kernels
        [value{m}, index{m}] = min(residual{m} , [], 2); 
    end
    [value_composite, index_composite] = min(residual_composite, [], 2);
    %%%%%%%%% record bool varible zi%%%%%%%%%%%%
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

    miss_choose_kernel_value = [];
    for i = 1:num_kernels
        miss_choose_kernel_value(i) = miss_choose_kernel_group(alignment_sort_index(i));%使准确率按照对齐分数顺序进行排序
    end
    mu = 0.02;
    [miss_choose_kernel_value_max,miss_choose_kernel_index_max] = max(miss_choose_kernel_value);%找到最大准确率所对应的核索引
    for i = 1:miss_choose_kernel_index_max
        if miss_choose_kernel_value(i) + mu > miss_choose_kernel_value_max %识别准确率
            miss_choose_kernel_index_selected = i;
            break;
        end
    end 
    choose_kernel_index = alignment_sort_index(miss_choose_kernel_index_selected);
    disp(choose_kernel_index)

    Wnewkernel = sum(double(choose_kernel{choose_kernel_index}&(1-current_kernel)))/sum(double((1-choose_kernel{choose_kernel_index})|(1-current_kernel)));
    Wcurrentkernel = sum(double((1-choose_kernel{choose_kernel_index})&current_kernel))/sum(double((1-choose_kernel{choose_kernel_index})|(1-current_kernel)));
  
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
    beta_group_error(t) = sqrt(sum((beta_group_previous - beta_group).^2));
    
      
end
% test
for m = 1:num_kernels
    YTY_kernel{m} = rand(dimension,dimension);
end
beta_group = rand(num_kernels,1);
YTY = zeros(dimension, dimension);  
for p = 1:num_kernels
    YTY = YTY + beta_group(p) * YTY_kernel{p};
end
for j = 1:length(tt_label_view)
    zTY_test = zeros(1, dimension);
    zTz_test = 0;
    for p = 1:num_kernels
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
    tic
    [x_test] = KOMP_ONE(0, zTY_test, YTY, A, dimension, sparsity);
    toc
    %Calculation of K(z, z)
    for i = 1:num_class       
        code_test = x_test(tr_label_view == i); %code is a coefficient vector 
        residual_test(j, i) = zTz_test - 2 * zTY_test(tr_label_view == i) * code_test + code_test' * YTY(tr_label_view == i, tr_label_view == i) * code_test;
      
    end
end

[value_test, index_test] = min(residual_test, [], 2);
correct_count_test = 0;
for i = 1:length(tt_label_view)
    if index_test(i) == tt_label_view(i)
        correct_count_test = correct_count_test + 1;
    end
end
accuracy = correct_count_test/length(tt_label_view);

%functions
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


      