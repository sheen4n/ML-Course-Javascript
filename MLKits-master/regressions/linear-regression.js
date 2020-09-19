const tf = require('@tensorflow/tfjs');

class LinearRegression {
  constructor(features, labels, options) {
    this.isInitialized = false;

    this.labels = tf.tensor(labels);
    this.mseHistory = [];
    // this.bHistory = [];
    this.features = this.processFeatures(features);

    this.weights = tf.zeros([this.features.shape[1], 1]);

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10,
      },
      options,
    );
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);
    const slopes = features.transpose().matMul(differences).div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchSize = this.options.batchSize;
    // const iterations = Math.floor(this.options.iterations / batchSize);
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);

        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
        this.gradientDescent(featureSlice, labelSlice);
      }
      this.recordMSE();
      this.updateLearningRate();
      // this.bHistory.push(this.weights.get(0, 0));
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    testLabels = tf.tensor(testLabels);
    testFeatures = this.processFeatures(testFeatures);

    const predictions = testFeatures.matMul(this.weights);

    const sumOfSquareOfResiduals = testLabels.sub(predictions).pow(2).sum().get();

    const sumOfSquares = testLabels.sub(testLabels.mean()).pow(2).sum().get();

    const coefficientOfDetermination = 1 - sumOfSquareOfResiduals / sumOfSquares;

    return coefficientOfDetermination;
  }

  generateMeanAndVariance(features) {
    if (this.isInitialized) return;

    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance;
    this.isInitialized = true;
  }

  processFeatures(features) {
    features = tf.tensor(features);
    this.generateMeanAndVariance(features);
    features = this.standardize(features);

    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standardize(features) {
    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;

    const lastValue = this.mseHistory[0];
    const secondLastValue = this.mseHistory[1];

    if (lastValue > secondLastValue) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;

// gradientDescent() {
//   const currentGuessesForMPG = this.features.map((row) => this.m * row[0] + this.b);

//   const bSlope =
//     2 *
//     (1 / this.features.length) *
//     currentGuessesForMPG
//       .map((guess, i) => guess - this.labels[i][0])
//       .reduce((acc, curr) => acc + curr, 0);

//   const mSlope =
//     2 *
//     (1 / this.features.length) *
//     currentGuessesForMPG
//       .map((guess, i) => -1 * (this.labels[i][0] - guess))
//       .reduce((acc, curr) => acc + curr, 0);

//   this.m = this.m - mSlope * this.options.learningRate;
//   this.b = this.b - bSlope * this.options.learningRate;
// }
