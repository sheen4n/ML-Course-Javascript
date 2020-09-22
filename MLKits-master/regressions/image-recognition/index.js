require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const MultinominalRegression = require('./multinominal-regression');
const mnist = require('mnist-data');

const n = 10;
const max = 1000;
const mnistData = mnist.training(0, max);
const features = mnistData.images.values.map((number) => number.flat());
const encodedLabels = mnistData.labels.values.map((v) =>
  Array.from(Array(n), (_, i) => (i === v ? 1 : 0)),
);

const regression = new MultinominalRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 5,
  batchSize: 100,
});

regression.train();

const testingSize = 100;
const testMnistData = mnist.testing(0, testingSize);
const testFeatures = testMnistData.images.values.map((number) => number.flat());
const testEncodedLabels = testMnistData.labels.values.map((v) =>
  Array.from(Array(n), (_, i) => (i === v ? 1 : 0)),
);

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log(accuracy);
