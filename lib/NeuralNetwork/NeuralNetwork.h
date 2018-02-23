#ifndef NEURALNET_H
#define NEURALNET_H


#include <Arduino.h>
// #define NDEBUG // uncomment for disabling assert()
#include <cassert>
#include <vector>
  using namespace std;




// ********************************* TYPEDEFS **********************************

typedef vector<double>   Table;
typedef vector<uint32_t> Topology;

class Neuron;
typedef vector<Neuron> Layer;

struct Connection;
typedef vector<Connection> Connections;

// *****************************************************************************




// ********************************** NEURON ***********************************

struct Connection { double weight, deltaWeight; };

class Neuron {

  static double eta, alpha;

  double      m_outputVal;
  Connections m_outputWeights;
  uint32_t    m_index;
  double      m_gradient;

  static double activation (double x) { return tanh(x); }
  static double activationD(double x) { return 1.0/(cosh(x)*cosh(x)); }

 public:

  Neuron(uint32_t, uint32_t);

  void   setOutput(double v)       { m_outputVal = v;    }
  double getOutput(void)     const { return m_outputVal; }

  void feedForward(const Layer &);
  void calcGrad(double);
  void calcHiddenGrad(const Layer &);
  void updateInputWeights(Layer &);
};

// *****************************************************************************




// ********************************* NETWORK ***********************************

class NeuralNetwork {

  vector<Layer> m_layers; // structure[layerNum][neuronNum]
  double m_error, m_rAvgError, m_rAvgSmoothing;

 public:

  NeuralNetwork(const Topology &);

  void feedForward(const Table &);
  void backProp   (const Table &);
  void getResults (      Table &) const;
};

// *****************************************************************************




#endif // NEURALNET_H
