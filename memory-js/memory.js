const _ = require('lodash');

// demonstrate how to utilize memory in node
const loadData = () => {
  const randoms = _.range(0, 999999);
  return randoms; // can be garbage collected
};

const data = loadData();
debugger;
