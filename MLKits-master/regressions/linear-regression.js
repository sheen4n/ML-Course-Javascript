const tf = require('@tensorflow/tfjs');

class LinearRegression {
  constructor(features, labels, options) {
    this.isInitialized = false;

    this.labels = tf.tensor(labels);
    this.features = this.processFeatures(features);

    this.weights = tf.zeros([2, 1]);

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
      },
      options
    );
  }

  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);
    const slopes = this.features.transpose().matMul(differences).div(this.features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
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
