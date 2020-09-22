const tf = require('@tensorflow/tfjs');

class MultinominalRegression {
  constructor(features, labels, options) {
    this.isInitialized = false;

    this.labels = tf.tensor(labels);
    this.costHistory = [];
    this.features = this.processFeatures(features);

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);

    this.options = Object.assign(
      {
        batchSize: 10,
        decisionBoundary: 0.5,
        iterations: 1000,
        learningRate: 0.1,
      },
      options,
    );
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const softmaxGuesses = currentGuesses.softmax();
    const differences = softmaxGuesses.sub(labels);
    const slopes = features.transpose().matMul(differences).div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchSize = this.options.batchSize;
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
        this.gradientDescent(featureSlice, labelSlice);
      }
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights).softmax().argMax(1);
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);
    const incorrect = predictions.notEqual(testLabels).sum().get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
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
    const mappedVariance = this.variance
      .cast('bool')
      .logicalNot()
      .cast('float32')
      .add(this.variance);

    this.variance = mappedVariance;
    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();
    const termOne = this.labels.transpose().matMul(guesses.log());
    const termTwo = this.labels.mul(-1).add(1).transpose().matMul(guesses.mul(-1).add(1).log());

    const cost = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);
    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) return;

    const lastValue = this.costHistory[0];
    const secondLastValue = this.costHistory[1];

    if (lastValue > secondLastValue) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = MultinominalRegression;
