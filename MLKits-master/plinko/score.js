const outputs = [];

const getAbsDistance = (item, compared) =>
  Math.sqrt(
    item
      .slice(0, item.length - 1)
      .map((_, i) => (item[i] - compared[i]) ** 2)
      .reduce((acc, curr) => acc + curr),
  );

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const length = 3;
  const range = Array.from({ length }, (_, i) => i + 1);

  const normalizedOutputs = normalizeData(outputs);

  const featuresCount = outputs[0].length - 1;

  for (let i = 0; i < featuresCount; i++) {
    const singleFeatureOutputs = normalizedOutputs.map((row) => [row[i], ...row.slice(-1)]);

    range.map((k) => {
      const testSetSize = 10;
      const [testSet, trainingSet] = splitDataset(singleFeatureOutputs, testSetSize);
      const matchedCount = testSet.filter(
        (testRow) => knn(trainingSet, testRow, k) === testRow.slice(-1)[0],
      ).length;

      console.log(`Feature ${i} - Accuracy for k : ${k} =  ${(matchedCount * 100) / testSetSize}%`);
    });
  }
}

// K Nearest Neighbour
function knn(data, testRow, k) {
  const resultsDict = data
    .map((item) => [getAbsDistance(item, testRow), ...item.slice(-1)])
    .sort((a, b) => a[0] - b[0])
    .slice(0, k)
    .reduce((acc, curr) => {
      acc[curr[1]] = acc[curr[1]] ? ++acc[curr[1]] : 1;
      return acc;
    }, {});

  const bucket = Object.entries(resultsDict).sort((a, b) => b[1] - a[1])[0][0];

  return Number(bucket);
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = shuffled.slice(0, testCount);
  const trainingSet = shuffled.slice(testCount);

  return [testSet, trainingSet];
}

const zip = (rows) => rows[0].map((_, c) => rows.map((row) => row[c]));

function normalizeData(data) {
  const normalizedAll = zip(data);

  const minMaxOfEachFeature = normalizedAll.slice(0, data[0].length - 1).map((feature) => ({
    max: Math.max(...feature),
    min: Math.min(...feature),
  }));

  return data.map((row, i) => {
    return row.map((val, i) => {
      if (i === data[0].length - 1) return val;
      if (minMaxOfEachFeature[i].max === minMaxOfEachFeature[i].min) return val;
      return (
        (val - minMaxOfEachFeature[i].min) /
        (minMaxOfEachFeature[i].max - minMaxOfEachFeature[i].min)
      );
    });
  });
}
