#ifndef NEURALNET_H
#define NEURALNET_H


#include "Arduino.h"
#include <vector>


//############################################################################//
//                     Default values for static variables                    //
//############################################################################//
//----------------------------------------------------------------------------//
  #define stdETA           0.15     // Neuron learning rate                   //
  #define stdALPHA         0.5      // Neuron learning momentum               //
  #define stdSMOOTHING    20.0      // # of samples to average rAvgError over //
//----------------------------------------------------------------------------//
//############################################################################//


//############################################################################//
//                                  TYPEDEFS                                  //
//############################################################################//
//----------------------------------------------------------------------------//
  class Neuron;                                                               //
  typedef std::vector<Neuron> Layer;                                          //
  typedef std::vector<Layer>  Net;                                            //
//----------------------------------------------------------------------------//
  typedef std::vector<double> List;                                           //
  typedef std::vector<List>   Table;                                          //
//----------------------------------------------------------------------------//
//############################################################################//




// ================================== NEURON ===================================

struct Connection { double weight, deltaWeight; };


class Neuron {

private:
  std::vector<Connection> _connections;
  
  double   _value;
  uint32_t _index;
  double   _gradient;

  // TODO: Pool of activation functions to choose from
  static double activation (double &x) { return tanh(x); }
  static double activationD(double  x) { x = cosh(x); return 1.0/(x*x); }

public:
  static double eta, alpha;           // eta: "learning rate", alpha: "momentum"

  Neuron(uint32_t, uint32_t);

  double getValue(void)     const { return _value; }
  void   setValue(double v)       { _value = v;    }

  List getWeights(void) const;
  void setWeights(List &);

  void feedForward   (const Layer &);
  void calcGrad      (double);
  void calcHiddenGrad(const Layer &);
  void updateWeights (Layer &);
};

// ---------------------------------- NEURON -----------------------------------




// ================================== NETWORK ==================================

class NeuralNetwork {

private:
  Net _network;                                       // layers[#layer][#neuron]

  std::vector<uint32_t> _topology;

  double        _error, _rAvgError = 0.5;
  static double _rAvgSmoothing;

public:
  NeuralNetwork(const std::vector<uint32_t> &);

  NeuralNetwork& begin(double);
  NeuralNetwork& begin(double, double);

  NeuralNetwork& feedForward(const List &);
  NeuralNetwork& propBack   (const List &);

  NeuralNetwork& train   (const Table &, uint32_t);
  NeuralNetwork& memorize(void);
  NeuralNetwork& recall  (const List &);

  std::vector<uint32_t> getTopology(void) const { return _topology; }
  
  double getError   (void) const { return _error;     }
  double getAvgError(void) const { return _rAvgError; }

  List getOutput(void) const;
};

// TODO: class NeuralNetwork_RBF : public NeuralNetwork

// ---------------------------------- NETWORK ----------------------------------


#endif // NEURALNET_H
