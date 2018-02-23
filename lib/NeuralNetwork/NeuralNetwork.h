#ifndef NEURALNET_H
#define NEURALNET_H


#include <Arduino.h>
#include <vector>
  using namespace std;


// -----------------------------------------------------------------------------
//                      Standard values for static variables
// -----------------------------------------------------------------------------
   #define stdETA           0.15       // Neuron learning rate
   #define stdALPHA         0.0        // Neuron learning momentum
   #define stdSMOOTHING    20.0        // # of samples to average rAvgError over
// -----------------------------------------------------------------------------



// ********************************* TYPEDEFS **********************************

struct Connection;
typedef vector<Connection>    Connections;

class Neuron;
typedef vector<Neuron>        Layer;

typedef vector<double>        Table;
typedef vector<vector<Table>> TrainingData;

// *****************************************************************************




// ********************************** NEURON ***********************************

struct Connection { double weight, deltaWeight; };

class Neuron {

  double      _value;
  Connections _connections;
  uint32_t    _index;
  double      _gradient;

  static double activation (double x) { return tanh(x); }
  static double activationD(double x) { return 1 - x*x; } // 1.0/(cosh(x)*cosh(x))

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

  vector<Layer> _layers;                              // layers[#layer][#neuron]
  double        _error, _rAvgError = 0.5;
  static double _rAvgSmoothing;

 public:

  NeuralNetwork(const vector<uint32_t> &);

  void begin(double);
  void begin(double, double);

  double getAvgError(void) const { return _rAvgError; }

  void feedForward(const Table &);
  void propBack   (const Table &);
  void train      (const TrainingData &, uint32_t);

  Table getOutput(void) const;
};

// *****************************************************************************




#endif // NEURALNET_H
