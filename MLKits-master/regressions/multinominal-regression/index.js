require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const MultinominalRegression = require('./multinominal-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg'],
  converters: {
    mpg: (value) => {
      const mpg = parseFloat(value);

      if (mpg < 15) return [1, 0, 0];
      if (mpg < 30) return [0, 1, 0];
      return [0, 0, 1];
    },
  },
});

labels = labels.flat();

const regression = new MultinominalRegression(features, labels, {
  learningRate: 0.5,
  iterations: 50,
  batchSize: 5,
});

regression.train();
const accuracy = regression.test(testFeatures, testLabels.flat());
console.log(accuracy);
