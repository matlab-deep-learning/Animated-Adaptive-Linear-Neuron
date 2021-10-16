% Download sample data in csv
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
csvFile = 'iris.csv';
websave(csvFile, url);

tbl = readtable('iris.csv');
tbl1 = tbl(1:100, {'Var1','Var3'});
tbl1.Properties.VariableNames{'Var1'} = 'SepalLength';
tbl1.Properties.VariableNames{'Var3'} = 'PetalLength';
tbl1.cls = [-1*ones(50,1); ones(50,1)];

writetable(tbl1, 'iris_data.csv');

delete('iris.csv');