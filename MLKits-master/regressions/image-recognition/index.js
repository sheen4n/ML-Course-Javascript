require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const MultinominalRegression = require('./multinominal-regression');
const mnist = require('mnist-data');

const n = 10;

const generateFeaturesAndLabels = (size) => {
  const mnistData = mnist.training(0, size);
  const features = mnistData.images.values.map((pixels) => pixels.flat());
  const encodedLabels = mnistData.labels.values.map((v) =>
    Array.from(Array(n), (_, i) => (i === v ? 1 : 0)),
  );
  return { features, labels: encodedLabels };
};

const featureSetSize = 60000;
const { features, labels } = generateFeaturesAndLabels(featureSetSize);

const testingSetSize = 20000;
const { features: testFeatures, labels: testEncodedLabels } = generateFeaturesAndLabels(
  testingSetSize,
);

const regression = new MultinominalRegression(features, labels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 50,
});
regression.train();

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log(accuracy);
