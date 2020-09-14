require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

const knn = (features, labels, predictionPoint, k) => {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
  const scaledFeatures = features.sub(mean).div(variance.pow(0.5));

  return (
    scaledFeatures
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => a.get(0) - b.get(0))
      .slice(0, k)
      .reduce((acc, curr) => acc + curr.get(1), 0) / k
  );
};

let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
  shuffle: true,
  splitTest: 10,
  dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
  labelColumns: ['price'],
});

testFeatures.map((testFeature, i) => {
  const result = knn(tf.tensor(features), tf.tensor(labels), tf.tensor(testFeature), 10);
  const err = (testLabels[i][0] - result) / testLabels[i][0];

  console.log(err * 100);
});
