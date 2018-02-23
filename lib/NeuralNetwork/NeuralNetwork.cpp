#include <NeuralNetwork.h>



// ********************************** NEURON ***********************************

double Neuron::eta   = 0.15;
double Neuron::alpha = 0.5;


Neuron::Neuron(uint32_t numOutputs, uint32_t index) {

  for (size_t connection = 0; connection < numOutputs; connection++) {

    m_outputWeights.push_back(Connection());
    m_outputWeights.back().weight = random(0, 1000)/1000.0;
  }

  m_index = index;
}


void Neuron::feedForward(const Layer &prevLayer) {

  double sum = 0.0;

  for (size_t n = 0; n < prevLayer.size(); n++) {
    sum += prevLayer[n].getOutput() * prevLayer[n].m_outputWeights[m_index].weight;
  }

  m_outputVal = Neuron::activation(sum);
}


void Neuron::calcGrad(double target) {

  m_gradient = (target - m_outputVal) * Neuron::activationD(m_outputVal);
}


void Neuron::calcHiddenGrad(const Layer &nextLayer) {

  double sum = 0.0;

  for (size_t n = 0; n < nextLayer.size() - 1; n++) {
    sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
  }

  m_gradient = sum * Neuron::activationD(m_outputVal);
}


void Neuron::updateInputWeights(Layer &prevLayer) {

  for (size_t n = 0; n < prevLayer.size(); n++) {

    Neuron &neuron = prevLayer[n];

    double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

    double newDeltaWeight =
      eta   * neuron.getOutput() * m_gradient +
      alpha * oldDeltaWeight;

    neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
    neuron.m_outputWeights[m_index].weight     += newDeltaWeight;
  }
}

// *****************************************************************************




// ********************************* NETWORK ***********************************

NeuralNetwork::NeuralNetwork(const Topology &topology) {

  uint32_t numLayers = topology.size();

  for (size_t layer = 0; layer < numLayers; layer++) {

    m_layers.push_back(Layer());

    uint32_t numOutputs;
    layer == topology.size() - 1
      ? numOutputs = 0
      : numOutputs = topology[layer + 1];

    for (size_t neuron = 0; neuron <= topology[layer]; neuron++) {
      m_layers.back().push_back(Neuron(numOutputs, neuron));
    }
  }
}


void NeuralNetwork::feedForward(const Table &input) {

  assert(input.size() == m_layers[0].size() - 1);

  for (size_t i = 0; i < input.size(); i++) {
    m_layers[0][i].setOutput(input[i]);
  }

  for (size_t layer = 0; layer < m_layers.size(); layer++) {

    Layer &prevLayer = m_layers[layer - 1];

    for (size_t n = 0; n < m_layers[layer].size() - 1; n++) {
      m_layers[layer][n].feedForward(prevLayer);
    }
  }
}


void NeuralNetwork::backProp(const Table &target) {

  // Calc overall network error (RMS)

  Layer &outputLayer = m_layers.back();
  m_error = 0.0;

  for (size_t n = 0; n < outputLayer.size() - 1; n++) {

    double delta = target[n] - outputLayer[n].getOutput();
    m_error += delta*delta;
  }

  m_error /= outputLayer.size() - 1;
  m_error  = sqrt(m_error);


  // Calc recent average error (rAvgError)

  m_rAvgError = (m_rAvgError * m_rAvgSmoothing + m_error) / (m_rAvgSmoothing + 1.0);


  // Calc gradients in output layer

  for (size_t n = 0; n < outputLayer.size() - 1; n++) {
    outputLayer[n].calcGrad(target[n]);
  }


  // Calc gradients in hidden layers

  for (size_t layer = m_layers.size() - 2; layer < 0; layer--) {

    Layer &hiddenLayer = m_layers[layer];
    Layer &nextLayer   = m_layers[layer + 1];

    for (size_t n = 0; n < hiddenLayer.size(); n++) {
      hiddenLayer[n].calcHiddenGrad(nextLayer);
    }
  }


  // Update connection weights from output layer to first hidden layer

  for (size_t l = m_layers.size() - 1; l > 0; l--) {

    Layer &layer     = m_layers[l];
    Layer &prevLayer = m_layers[l - 1];

    for (size_t n = 0; n < layer.size() - 1; n++) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}


void NeuralNetwork::getResults(Table &result) const {

  result.clear();

  for (size_t n = 0; n < m_layers.back().size() - 1; n++) {
    result.push_back(m_layers.back()[n].getOutput());
  }
}

// *****************************************************************************
