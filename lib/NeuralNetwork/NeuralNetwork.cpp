#include <NeuralNetwork.h>



// ********************************** NEURON ***********************************

double Neuron::eta   = stdETA;
double Neuron::alpha = stdALPHA;


Neuron::Neuron(uint32_t numOutputs, uint32_t index) {

  for (size_t connection = 0; connection < numOutputs; connection++) {

    _connections.push_back(Connection());
    _connections.back().weight = random(0, 1000)/1000.0;
  }

  _index = index;
}


Table Neuron::getWeights() const {

  Table weights;

  for (size_t w = 0; w < _connections.size(); w++) {
    weights.push_back(_connections[w].weight);
  }

  return weights;
}


void Neuron::setWeights(Table &weights) {

  for (size_t w = 0; w < _connections.size(); w++) {
    _connections[w].weight = weights[w];
  }
}


void Neuron::feedForward(const Layer &prevLayer) {

  double sum = 0.0;

  for (size_t neuron = 0; neuron < prevLayer.size(); neuron++) {
    sum += prevLayer[neuron].getValue() * prevLayer[neuron]._connections[_index].weight;
  }

  _value = Neuron::activation(sum);
}


void Neuron::calcGrad(double target) {

  _gradient = (target - _value) * Neuron::activationD(_value);
}


void Neuron::calcHiddenGrad(const Layer &nextLayer) {

  double sumDOW = 0.0;

  for (size_t n = 0; n < nextLayer.size() - 1; n++) {
    sumDOW += _connections[n].weight * nextLayer[n]._gradient;
  }

  _gradient  = sumDOW * Neuron::activationD(_value);
}


void Neuron::updateWeights(Layer &prevLayer) {

  for (size_t n = 0; n < prevLayer.size(); n++) {

    Neuron &neuron = prevLayer[n];

    double oldDeltaWeight = neuron._connections[_index].deltaWeight;

    double newDeltaWeight =
      eta   * neuron.getValue() * _gradient +
      alpha * oldDeltaWeight;

    neuron._connections[_index].deltaWeight = newDeltaWeight;
    neuron._connections[_index].weight     += newDeltaWeight;
  }
}

// *****************************************************************************




// ********************************* NETWORK ***********************************

double NeuralNetwork::_rAvgSmoothing = stdSMOOTHING;


NeuralNetwork::NeuralNetwork(const vector<uint32_t> &topology) {

  uint32_t numLayers = topology.size();

  for (size_t l = 0; l < numLayers; l++) {

    _layers.push_back(Layer());

    uint32_t numOutputs;
    l == topology.size() - 1
      ? numOutputs = 0
      : numOutputs = topology[l + 1];

    for (size_t n = 0; n <= topology[l]; n++) {
      _layers.back().push_back(Neuron(numOutputs, n));
    }

    // Set value of bias neuron to 1.0
    _layers.back().back().setValue(1.0);
  }

  _topology = topology;
}


void NeuralNetwork::begin(double learningRate) {

  Neuron::eta = learningRate;
}


void NeuralNetwork::begin(double learningRate, double momentum) {

  Neuron::eta   = learningRate;
  Neuron::alpha = momentum;
}


void NeuralNetwork::feedForward(const Table &input) {

  for (size_t n = 0; n < input.size(); n++) {
    _layers[0][n].setValue(input[n]);
  }

  for (size_t l = 1; l < _layers.size(); l++) {

    Layer &prevLayer = _layers[l - 1];

    for (size_t n = 0; n < _layers[l].size() - 1; n++) {
      _layers[l][n].feedForward(prevLayer);
    }
  }
}


void NeuralNetwork::propBack(const Table &target) {

  // Calc overall network error (RMS)

  Layer &outputLayer = _layers.back();
  _error = 0.0;

  for (size_t n = 0; n < outputLayer.size() - 1; n++) {

    double delta = target[n] - outputLayer[n].getValue();
    _error += delta * delta;
  }

  _error /= outputLayer.size() - 1;
  _error  = sqrt(_error);


  // Calc recent average error (rAvgError)

  _rAvgError = (_rAvgError * _rAvgSmoothing + _error) / (_rAvgSmoothing + 1.0);


  // Calc gradients in output layer

  for (size_t n = 0; n < outputLayer.size() - 1; n++) {
    outputLayer[n].calcGrad(target[n]);
  }


  // Calc gradients in hidden layers

  for (size_t l = _layers.size() - 2; l > 0; l--) {

    Layer &hiddenLayer = _layers[l];
    Layer &nextLayer   = _layers[l + 1];

    for (size_t n = 0; n < hiddenLayer.size(); n++) {
      hiddenLayer[n].calcHiddenGrad(nextLayer);
    }
  }


  // Update connection weights

  for (size_t l = _layers.size() - 1; l > 0; l--) {

    Layer &layer     = _layers[l];
    Layer &prevLayer = _layers[l - 1];

    for (size_t n = 0; n < layer.size() - 1; n++) {
      layer[n].updateWeights(prevLayer);
    }
  }
}


void NeuralNetwork::train(const TrainingData &data, uint32_t numRuns) {

  Serial.println("-- TRAINING SESSION --"); Serial.println();

  for (size_t run = 0; run < 10; run++) {

    // Progress indicator --------------------------------------

    Serial.print("  [");
    for (size_t i = 0;      i < run; i++) { Serial.print("#"); }
    for (size_t i = 10-run; i > 0;   i--) { Serial.print("_"); }
    Serial.print("]");

    run == 0
      ? Serial.print("   ")
      : Serial.print("  ");

    Serial.print(10*run);
    Serial.println(" %");

    // ---------------------------------------------------------

    for (size_t run = 0; run < numRuns/10; run++) {

      uint32_t num = random(0, data.size()/2);

      Table input  = data[2*num];
      Table target = data[2*num + 1];

      // Training sample
      feedForward(input);
      propBack(target);

      #ifdef ESP8266
        yield();
      #endif
    }
  }

  Serial.println("  [##########] 100 %"  );
  Serial.println(                        );
  Serial.println("-------- DONE --------");
  Serial.print  ("    PASSES = "         ); Serial.println(numRuns      );
  Serial.print  ("     ERROR = "         ); Serial.println(_error,     5);
  Serial.print  (" rAvgERROR = "         ); Serial.println(_rAvgError, 5);
  Serial.println("----------------------");
  Serial.println(                        );
}


void NeuralNetwork::memorize() {

  // Collect weight of every neuronal connection in the network

  Table memory;

  for (size_t l = 0; l < _layers.size() - 1; l++) {

    for (size_t n = 0; n < _layers[l].size(); n++) {    // including bias neuron

      Neuron   &neuron    = _layers[l][n];
      uint32_t numWeights = neuron.getWeights().size();

      for (size_t w = 0; w < numWeights; w++) {
        memory.push_back(neuron.getWeights()[w]);
      }
    }
  }


  // Write weights & topology to serial port

  Serial.print("{ ");

  for (size_t i = 0; i < memory.size(); i++) {

    Serial.print(memory[i], 5);

    i == memory.size() - 1
      ? Serial.print(" ")
      : Serial.print(", ");

    #ifdef ESP8266
      yield();
    #endif
  }

  Serial.print("}; // topology = { ");

  for (size_t i = 0; i < _topology.size(); i++) {

    Serial.print(_topology[i]);

    i == _topology.size() - 1
      ? Serial.print(" ")
      : Serial.print(", ");
  }

  Serial.println("}");
}


void NeuralNetwork::recall(const Table &memory) {

  uint32_t tempIndex = 0;

  for (size_t l = 0; l < _layers.size() - 1; l++) {

    for (size_t n = 0; n < _layers[l].size(); n++) {    // including bias neuron

      Neuron   &neuron    = _layers[l][n];
      uint32_t numWeights = neuron.getWeights().size();

      Table weights;

      for (size_t w = 0; w < numWeights; w++) {

        weights.push_back(memory[tempIndex]);
        tempIndex++;
      }

      neuron.setWeights(weights);
    }
  }
}


Table NeuralNetwork::getOutput() const {

  Table output;

  for (size_t n = 0; n < _layers.back().size() - 1; n++) {

    double value = _layers.back()[n].getValue();
    output.push_back(value);
  }

  return output;
}

// *****************************************************************************
