function [out] = ps3(in)
%[min_cost_d, pred_error] = ps3_1

out = ps3_3

end

function [a, b] = ps3_1(in)
x_train = importdata('Xtrain.txt');
y_train = importdata('Ytrain.txt');
x_test = importdata('Xtest.txt');
y_test = importdata('Ytest.txt');

L = 0;

A = {};
beta_k = {};
for d = 1:10
    for i = 1:size(x_train,1)
        for j = 1:d
            A{d}(i,j) = x_train(i) ^ j ;
        end
    end
    beta_k{d} = A{d} \ y_train;
end


L = [];
y_hat  = [];
for d = 1:10
    L(d) = 0;
    for i = 1:size(x_train,1)
        inner_sum = 0;
        for k = 1:d
            inner_sum = inner_sum + beta_k{d}(k) * x_train(i) ^ k;
        end
        L(d) = L(d) + (y_train(i) - inner_sum) ^ 2;
    end
    y_hat(:,d) = A{d} * beta_k{d};
    
    
end

figure;
hold on
title('Regression Plots for varying degree polynomials');
plot (x_train, y_hat);
legend('d = 1', 'd = 2', 'd = 3', 'd = 4');
plot (x_train, y_train, 'k.');
hold off

beta_test = {};

A_test = {};
beta_k = {};
count = 1;
for d = [3, 10]
    for i = 1:size(x_test,1)
        for j = 1:d
            A_test{count}(i,j) = x_test(i) ^ j ;
        end
    end
    beta_test{end + 1} = A_test{end} \ y_test;
    count = count + 1;
end
pred_error = zeros(1,2);
y_hat = {};
for j = 1: size(beta_test,2)
    for i = 1:size(y_test,1)
        
        y_hat{j}(i) =  A_test{j}(i, :) * beta_test{j};
        pred_error(j) = pred_error(j) + (y_hat{j}(i) - y_test(i)) ^ 2;
    end
end
[min_L, idx] = min(L);

a = idx;
b = pred_error;
end

function [out] = ps3_2(in)

mu_i = [1,1];
mu_ii = [-1,2];
mu_iii = [0,2;2,0];
mu_iv = [0,2;2,0];
mu_v = [1,1;-1,-1];


sigma_i = [2,0;0,1];
sigma_ii = [3,1;1,2];
sigma_iii = [1,1;1,2];
sigma_iv_1 = [1,1;1,2];
sigma_iv_2 = [3,1;1,2];
sigma_v_1 = [1,0;0,2];
sigma_v_2 = [2,1;1,2];


x1 = -3:.2:3; x2 = -3:.2:3;
[X1,X2] = meshgrid(x1,x2);
y_1 = mvnpdf([X1(:),X2(:)], mu_i, sigma_i);
y_1 = reshape(y_1,length(x2),length(x1));
y_2 = mvnpdf([X1(:),X2(:)], mu_ii, sigma_ii);
y_2 = reshape(y_2,length(x2),length(x1));

y_3 = mvnpdf([X1(:),X2(:)], mu_iii(1,:), sigma_iii) - mvnpdf([X1(:),X2(:)], mu_iii(2,:), sigma_iii);
y_3 = reshape(y_3,length(x2),length(x1));figure;

y_4 = mvnpdf([X1(:),X2(:)], mu_iv(1,:), sigma_iv_1) - mvnpdf([X1(:),X2(:)], mu_iv(2,:), sigma_iv_2);
y_4 = reshape(y_4,length(x2),length(x1));figure;


y_5 = mvnpdf([X1(:),X2(:)], mu_v(1,:), sigma_v_1) - mvnpdf([X1(:),X2(:)], mu_v(2,:), sigma_v_2);
y_5 = reshape(y_5,length(x2),length(x1));


figure;
surf(x1,x2,y_1);

figure;
surf(x1,x2,y_2);

figure;
surf(x1,x2,y_3);

figure;
surf(x1,x2,y_4);

figure;
surf(x1,x2,y_5);

end


function [out] = ps3_3(in)

D = load('train_small.mat');
T = load('test.mat');

cov_X = cell(1,7);
mean_X = cell(1,7);
overall_cov = cell(1,7);
for i = 1:7
    overall_cov{i} = zeros(784, 784);
    cov_X{i} = cell(1,10);
    mean_X{i} = cell(1,10);
    images = im2double(D.train{i}.images);
    D_labels = D.train{i}.labels;
    
    D_im = reshape(permute(images,[3,1,2]), size(images,3), 784);
    X = cell(1, 10);
    for y = 1:size(D_labels,1)
        X{D_labels(y) + 1} = [X{D_labels(y) + 1}; D_im(y,:)];
    end
    
    for z = 1:size(X, 2)
        cov_X{i}{z} = cov(double(X{z}));
        mean_X{i}{z} = double(sum(X{z})) / double(size(X{z}, 1));
        overall_cov{i} = overall_cov{i} +  cov_X{i}{z};
    end
    overall_cov{i} = .1 * overall_cov{i};
    %         for w = 1:size(cov_X{i}, 2)
    %             figure;
    %             imagesc(cov_X{i}{w});
    %             set(gca, 'XTick', [-1:1]);
    %             set(gca, 'YTick', [-1:1]);
    %             xlabel('Predicted Label');
    %             ylabel('True Label');
    %             colorbar
    %         end
    %         figure;
    %         imagesc(overall_cov{i});
    %         set(gca, 'XTick', [-1:1]);
    %         set(gca, 'YTick', [-1:1]);
    %         xlabel('Predicted Label');
    %         ylabel('True Label');
    %         colorbar
    
end
T_images = im2double(T.test.images);
T_labels = T.test.labels;
T_im_vect = [];
T_im_vect = reshape(permute(T_images,[3,1,2]),size(T_images,3), 784);
alpha = .3;
error_a = zeros(1,7);
error_b = zeros(1,7);

for i = 1: 7
    Y_a = zeros(size(T_im_vect, 1), 10);
    Y_b =  zeros(size(T_im_vect, 1), 10);
    for j = 1:10
        Y_a(:, j) = mvnpdf(double(T_im_vect), mean_X{i}{j}, overall_cov{i} + alpha * eye(784));
        Y_b(:, j) = mvnpdf(double(T_im_vect), mean_X{i}{j}, cov_X{i}{j} + alpha * eye(784));
    end

    [probability, idx_a] = max(Y_a');
    [probability, idx_b] = max(Y_b');
    pred_a = idx_a' - 1;
    pred_b = idx_b' - 1;
    error_a(i) = 1 -   double(sum(pred_a == T_labels)) / double(size(T_labels, 1))
    error_b(i) =  1 -   double(sum(pred_b == T_labels)) / double(size(T_labels, 1))
    
end

out = {error_a, error_b};
sizes = [100, 200, 500, 1000, 2000, 5000, 10000];

figure;
hold on
title('Learning curves for different sized training sets');
plot(sizes, error_a,'r');
plot(sizes, error_b,'b');
xlabel('Training set size');
ylabel('Error rate');
legend('Error rate for average covariance', 'Error rate for class specific covariance');

hold off

end