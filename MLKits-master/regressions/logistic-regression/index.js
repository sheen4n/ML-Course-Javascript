require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');

// const LogisticRegression = require('./logistic-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['passedemissions'],
  converters: {
    passedemissions: (value) => (value === 'TRUE' ? 1 : 0),
  },
});

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 20,
  batchSize: 10,
});

regression.train();

plot({
  x: regression.costHistory.reverse(),
});
