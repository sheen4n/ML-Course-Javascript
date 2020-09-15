require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LinearRegression = require('./linear-regression');

const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower'],
  labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.0001,
  iterations: 100,
});

// regression.features.print();
regression.train();

const r2 = regression.test(testFeatures, testLabels);

console.log(r2);
