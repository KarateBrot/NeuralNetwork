#include <NeuralNetwork.h>



// ********************************** NEURON ***********************************

double Neuron::eta   = 0.15;
double Neuron::alpha = 0.5;


Neuron::Neuron(uint32_t numOutputs, uint32_t index) {

  for (size_t connection = 0; connection < numOutputs; connection++) {

    _connections.push_back(Connection());
    _connections.back().weight = random(0, 1000)/1000.0;
  }

  _index = index;
}


void Neuron::feedForward(const Layer &prevLayer) {

  double sum = 0.0;

  for (size_t n = 0; n < prevLayer.size(); n++) {
    sum += prevLayer[n].getValue() * prevLayer[n]._connections[_index].weight;
  }

  _value = Neuron::activation(sum);
}


void Neuron::calcGrad(double target) {

  _gradient = (target - _value) * Neuron::activationD(_value);
}


void Neuron::calcHiddenGrad(const Layer &nextLayer) {

  double sum = 0.0;

  for (size_t n = 0; n < nextLayer.size() - 1; n++) {
    sum += _connections[n].weight * nextLayer[n]._gradient;
  }

  _gradient = sum * Neuron::activationD(_value);
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
}


void NeuralNetwork::begin(double learningRate, double momentum) {

  Neuron::eta   = learningRate;
  Neuron::alpha = momentum;
}


void NeuralNetwork::feedForward(const vector<double> &input) {

  for (size_t i = 0; i < input.size(); i++) {
    _layers[0][i].setValue(input[i]);
  }

  for (size_t l = 0; l < _layers.size(); l++) {

    Layer &prevLayer = _layers[l - 1];

    for (size_t n = 0; n < _layers[l].size() - 1; n++) {
      _layers[l][n].feedForward(prevLayer);
    }
  }
}


void NeuralNetwork::propBack(const vector<double> &target) {

  // Calc overall network error (RMS)

  Layer &outputLayer = _layers.back();
  _error = 0.0;

  for (size_t n = 0; n < outputLayer.size() - 1; n++) {

    double delta = target[n] - outputLayer[n].getValue();
    _error += delta*delta;
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

  for (size_t l = _layers.size() - 2; l < 0; l--) {

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


vector<double> NeuralNetwork::getOutput() const {

  vector<double> results;

  for (size_t n = 0; n < _layers.back().size() - 1; n++) {
    results.push_back(_layers.back()[n].getValue());
  }

  return results;
}

// *****************************************************************************
