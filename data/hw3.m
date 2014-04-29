train_small = load('data/train_small.mat');
avg_cov = cell(7,1);
all_covariances = cell(7,1);
all_averages = cell(7,1);
all_priors = cell(7,1);

for i=1:7,
	labels = train_small.train{i}.labels;
	images = im2double(train_small.train{i}.images);

	covariances = cell(10,1);
	averages = cell(10,1);
	priors = cell(10,1);

	sum_of_cov = 0;

	for j=0:9,
		indices = find(labels == j);
		total = 0;
		z = [];
		for index = transpose(indices),
			img = reshape(images(:,:,index),1,784);
			total = total + img;
			z = [z; img];
		end
		cov_matrix = cov(double(z));
		avg = double(total) / length(indices);
		sum_of_cov = sum_of_cov + cov_matrix;
		covariances{j+1} = cov_matrix;
		averages{j+1} = avg;
		priors{j+1} = sum(labels == j) / length(labels);  %for problem 3.2
	end
	avg_cov{i} = sum_of_cov / 10;
	all_covariances{i} = covariances;
	all_averages{i} = averages;
	all_priors{i} = priors;
end

test_data = load('data/test.mat');
test_labels = test_data.test.labels;
test_images = im2double(test_data.test.images);
error_rates_a = zeros(1,7);
error_rates_b = zeros(1,7);
pictures = zeros(10000,784);

for k=1:length(test_labels),
    pictures(k,:) = reshape(test_images(:,:,k),1,784);
end

for i=1:7,
    predictions_a = zeros(1,length(test_labels));
    predictions_b = zeros(1,length(test_labels));
    probabilities_a = [];
    probabilities_b = [];
    for j=0:9,
        likelihood_a = mvnpdf(pictures, all_averages{i}{j+1},avg_cov{i}+eye(784)*0.025);
        likelihood_b = mvnpdf(pictures, all_averages{i}{j+1},all_covariances{i}{j+1}+eye(784)*0.025);
        prior = all_priors{i}{j+1};
        probabilities_a = [probabilities_a; transpose(likelihood_a * prior)];
        probabilities_b = [probabilities_b; transpose(likelihood_b * prior)];
    end
    [maxVal_a  maxInd_a] = max(probabilities_a);
    [maxVal_b  maxInd_b] = max(probabilities_b);
    predictions_a = transpose(maxInd_a - 1);
    predictions_b = transpose(maxInd_b - 1);
    error_rates_a(i) = 1 - sum(predictions_a == test_labels) / length(test_labels);
    error_rates_b(i) = 1 - sum(predictions_b == test_labels) / length(test_labels);
end
error_rates_a
error_rates_b

x_axis = [100 200 500 1000 2000 5000 10000];
bar(x_axis,error_rates_a)
bar(x_axis,error_rates_b)