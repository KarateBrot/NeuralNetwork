#include "NeuralNetwork.h"


// ================================== NEURON ===================================

double Neuron::eta   = stdETA;
double Neuron::alpha = stdALPHA;


Neuron::Neuron(uint32_t numOutputs, uint32_t index) {

  // Allocate memory for _connections
  _connections.reserve(numOutputs);

  for (size_t connection = 0; connection < numOutputs; connection++) {

    // Add new connection
    _connections.emplace_back();
    _connections.back().weight = rand()/(double)RAND_MAX;
  }

  _index = index;
}


List Neuron::getWeights() const {

  List weights;

  // Number of weights
  uint32_t numWeights = _connections.size();

  // Allocate memory for weights
  weights.reserve(numWeights);

  for (size_t w = 0; w < numWeights; w++) {
    weights.push_back(_connections[w].weight);
  }

  return weights;
}


void Neuron::setWeights(List &weights) {

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

// ---------------------------------- NEURON -----------------------------------




// ================================== NETWORK ==================================

double NeuralNetwork::_rAvgSmoothing = stdSMOOTHING;


NeuralNetwork::NeuralNetwork(const std::vector<uint32_t> &topology) :

  _topology(topology) {

  uint32_t numLayers = _topology.size();

  // Allocate memory for _network
  _network.reserve(numLayers);

  for (size_t l = 0; l < numLayers; l++) {

    // Add new layer
    _network.emplace_back();

    // Get number of neurons (incl. bias neuron) in current layer
    uint32_t numNeurons = _topology[l] + 1;

    // Allocate memory for current layer
    _network.back().reserve(numNeurons);

    // Get number of neural outputs for neurons in current layer
    uint32_t numOutputs;
    l == _topology.size() - 1
      ? numOutputs = 0
      : numOutputs = _topology[l + 1];

    // Give RNG seed a runtime dependent value to (hopefully) introduce non-reproductiveness
    srand(micros());

    // Add neurons to current layer
    for (size_t n = 0; n < numNeurons; n++) {
      _network.back().emplace_back(numOutputs, n);
    }

    // Set value of bias neuron to 1.0
    _network.back().back().setValue(1.0);
  }
}


NeuralNetwork& NeuralNetwork::begin(double learningRate) {

  Neuron::eta = learningRate;

  return *this;
}


NeuralNetwork& NeuralNetwork::begin(double learningRate, double momentum) {

  Neuron::eta   = learningRate;
  Neuron::alpha = momentum;

  return *this;
}


NeuralNetwork& NeuralNetwork::feedForward(const List &input) {

  for (size_t n = 0; n < input.size(); n++) {
    _network[0][n].setValue(input[n]);
  }

  for (size_t l = 1; l < _network.size(); l++) {

    Layer &prevLayer = _network[l - 1];

    for (size_t n = 0; n < _network[l].size() - 1; n++) {
      _network[l][n].feedForward(prevLayer);
    }
  }

  return *this;
}


NeuralNetwork& NeuralNetwork::propBack(const List &target) {

  // Calc overall network error (RMS)
  Layer &outputLayer = _network.back();
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
  for (size_t l = _network.size() - 2; l > 0; l--) {

    Layer &hiddenLayer = _network[l];
    Layer &nextLayer   = _network[l + 1];

    for (size_t n = 0; n < hiddenLayer.size(); n++) {
      hiddenLayer[n].calcHiddenGrad(nextLayer);
    }
  }

  // Update connection weights
  for (size_t l = _network.size() - 1; l > 0; l--) {

    Layer &layer     = _network[l];
    Layer &prevLayer = _network[l - 1];

    for (size_t n = 0; n < layer.size() - 1; n++) {
      layer[n].updateWeights(prevLayer);
    }
  }

  return *this;
}


NeuralNetwork& NeuralNetwork::train(const Table &data, uint32_t numRuns) {

  Serial.println( "----------------------------" );
  Serial.println( "----- TRAINING SESSION -----" );
  Serial.println( "----------------------------" );
  Serial.println(                                );
  Serial.println( "   0%       50%      100%"    );
  Serial.print  ( "   ["                         );

  uint32_t duration = millis();

  // Training --------------------------------------

  for (size_t run = 0; run < 20; run++) {

    for (size_t run = 0; run < numRuns/20; run++) {

      // Random number between 0 and data.size/2 (exclusively)
      uint32_t num = (uint32_t)(rand()/(double)RAND_MAX * (data.size()/2 -1) + 0.5);

      List input  = data[2*num];
      List target = data[2*num + 1];

      // Training sample
      feedForward(input);
      propBack(target);

      #ifdef ESP8266
        yield();
      #endif
    }

    Serial.print("#");                                     // Progress indicator
  }

  // -----------------------------------------------

  duration = millis() - duration;

  uint32_t min =  duration/60000;
  double   sec = (duration%60000)/1000.0;

  Serial.println( "]"                            );
  Serial.println( "            DONE"             );
  Serial.println(                                );
  Serial.println( "----------------------------" );
  Serial.print  ( "      PASSES = "              ); Serial.println(numRuns);
  Serial.print  ( "    DURATION = "              ); Serial.print  (min); Serial.print("m "); Serial.print(sec, 1); Serial.println("s");
  Serial.print  ( "        RATE = "              ); Serial.print  (numRuns*1000/(double)duration); Serial.println("/s");
  Serial.println( "----------------------------" );
  Serial.print  ( "       ERROR = "              ); Serial.println(_error,     6);
  Serial.print  ( "   rAvgERROR = "              ); Serial.println(_rAvgError, 6);
  Serial.println( "----------------------------" );
  Serial.println(                                );

  return *this;
}


NeuralNetwork& NeuralNetwork::memorize() {

  // Collect weight of every neuronal connection in the network
  List memory;

  for (size_t l = 0; l < _network.size() - 1; l++) {

    for (size_t n = 0; n < _network[l].size(); n++) {   // including bias neuron

      Neuron &neuron = _network[l][n];
      List   weights = neuron.getWeights();

      for (size_t w = 0; w < weights.size(); w++) {
        memory.push_back(weights[w]);
      }
    }
  }

  // Write memory size, topology & weights to serial port
  Serial.print("{  //  ");                                        // Memory size
  Serial.print(memory.size()*64/8000.0);
  Serial.print(" kB  |  ");

  Serial.print("topology = { ");                                     // Topology

  for (size_t i = 0; i < _topology.size(); i++) {

    Serial.print(_topology[i]);

    i == _topology.size() - 1
      ? Serial.print(" ")
      : Serial.print(", ");
  }

  Serial.println("}" );
  Serial.println(    );
  Serial.print  ("  ");

  for (size_t i = 0; i < memory.size(); i++) {                        // Weights

    if (memory[i] > 0.0) { Serial.print(" "); }

    Serial.print(memory[i], 5);

    i == memory.size() - 1
      ? Serial.print(" ")
      : Serial.print(", ");

    if (((i+1)) % 8 == 0 && i != 0) { Serial.println(); Serial.print("  "); }

    #ifdef ESP8266
      yield();
    #endif
  }

  Serial.println(); Serial.println("};");

  return *this;
}


NeuralNetwork& NeuralNetwork::recall(const List &memory) {

  uint32_t tempIndex = 0;

  for (size_t l = 0; l < _network.size() - 1; l++) {

    for (size_t n = 0; n < _network[l].size(); n++) {   // including bias neuron

      Neuron   &neuron    = _network[l][n];
      uint32_t numWeights = neuron.getWeights().size();

      List weights;

      // Allocate memory for weights
      weights.reserve(numWeights);

      for (size_t w = 0; w < numWeights; w++) {
        weights.push_back(memory[tempIndex]);
        tempIndex++;
      }

      neuron.setWeights(weights);
    }
  }

  return *this;
}


List NeuralNetwork::getOutput() const {

  List output;

  // Number of neurons in output layer (excl. bias neuron)
  uint32_t numOutputs = _network.back().size() - 1;

  // Allocate memory for output
  output.reserve(numOutputs);

  for (size_t n = 0; n < numOutputs; n++) {
    double value = _network.back()[n].getValue();
    output.push_back(value);
  }

  return output;
}

// ---------------------------------- NETWORK ----------------------------------
