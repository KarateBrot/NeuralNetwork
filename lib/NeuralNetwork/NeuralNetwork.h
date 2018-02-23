#ifndef NEURALNET_H
#define NEURALNET_H


#include <Arduino.h>
#include <vector>
  using namespace std;




// ********************************* TYPEDEFS **********************************

struct Connection;
typedef vector<Connection> Connections;

class Neuron;
typedef vector<Neuron> Layer;

// *****************************************************************************




// ********************************** NEURON ***********************************

struct Connection { double weight, deltaWeight; };

class Neuron {

  double      _value;
  Connections _connections;
  uint32_t    _index;
  double      _gradient;

  static double activation (double x) { return tanh(x); }
  static double activationD(double x) { return 1.0/(cosh(x)*cosh(x)); }

 public:

  static double eta, alpha;           // eta: "learning rate", alpha: "momentum"

  Neuron(uint32_t, uint32_t);

  void   setValue(double v)       { _value = v;    }
  double getValue(void)     const { return _value; }

  void feedForward   (const Layer &);
  void calcGrad      (double);
  void calcHiddenGrad(const Layer &);
  void updateWeights (Layer &);
};

// *****************************************************************************




// ********************************* NETWORK ***********************************

class NeuralNetwork {

  vector<Layer> _layers; // layers[#layer][#neuron]
  double _error, _rAvgError, _rAvgSmoothing;

 public:

  NeuralNetwork(const vector<uint32_t> &);

  void begin(double, double);

  void feedForward(const vector<double> &);
  void propBack   (const vector<double> &);

  vector<double> getOutput(void) const;
};

// *****************************************************************************




#endif // NEURALNET_H
