#include <NeuralNetwork.h>

// XOR gate
TrainingData trainingData_xor({

  // input target
  { {0, 0}, {0} },
  { {0, 1}, {1} },
  { {1, 0}, {1} },
  { {1, 1}, {0} }

});
