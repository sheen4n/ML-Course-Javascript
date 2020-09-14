const tf = require('@tensorflow/tfjs');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = tf.tensor(features);
    this.labels = tf.tensor(labels);

    this.features = tf.ones([this.features.shape[0], 1]).concat(this.features, 1);
    this.weights = tf.zeros([2, 1]);

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
      },
      options,
    );
  }

  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);
    const slopes = this.features.transpose().matMul(differences).div(this.features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

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

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegression;
