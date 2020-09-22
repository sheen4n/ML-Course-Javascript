const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

function extractColumns(data, columnNames) {
  const headers = data[0];
  const indexes = columnNames.map((column) => headers.indexOf(column)); // return index of columnNames in header;
  const extracted = data.map((row) => _.pullAt(row, indexes));
  return extracted;
}

const seedPhrase = 'exampleSeedPhrase'; // configurable

function loadCSV(
  filename,
  { converters = {}, dataColumns = [], labelColumns = [], shuffle = true, splitTest = false },
) {
  let data = fs.readFileSync(filename, { encoding: 'utf-8' });
  data = data.split('\n').map((row) => row.split(','));
  data = data.map((row) => _.dropRightWhile(row, (val) => val === ''));

  const headers = data[0];

  data = data.map((row, index) => {
    if (index === 0) return row;

    return row.map((element, index) => {
      // reading through each cell
      const columnName = headers[index];

      if (converters[columnName]) {
        // check if columnName exists on converters
        const converted = converters[columnName](element); // pass into the function of converters , e.g. passed = val => val === "true"
        return !_.isNaN(converted) ? converted : element; // return converted if conversion success, else return element
      }

      const mappedNumber = Number(element); // attempt to convert to number
      return mappedNumber || mappedNumber === 0 ? mappedNumber : element; // return mappedNum if mapping success, else return element
    });
  });

  // finish transformation

  // extract label first
  let labels = extractColumns(data, labelColumns);
  labels.shift(); // remove the headers

  // narrow down data by keeping only required columns
  data = extractColumns(data, dataColumns);
  data.shift(); // remove the headers

  if (shuffle) {
    data = shuffleSeed.shuffle(data, seedPhrase);
    labels = shuffleSeed.shuffle(labels, seedPhrase);
  }

  if (splitTest) {
    const testSize = _.isNumber(splitTest) ? splitTest : Math.floor(data.length / 2);

    return {
      features: data.slice(0, testSize),
      labels: labels.slice(0, testSize),
      testFeatures: data.slice(testSize),
      testLabels: labels.slice(testSize),
    };
  } else {
    return {
      features: data,
      labels,
    };
  }
}

// loadCSV('data.csv');
const { features, labels, testFeatures, testLabels } = loadCSV('data.csv', {
  dataColumns: ['height', 'value'],
  labelColumns: ['passed'],
  converters: {
    passed: (val) => val === 'TRUE',
  },
  shuffle: true,
  splitTest: 1,
});

console.log(features);
console.log(labels);
console.log(testFeatures);
console.log(testLabels);
