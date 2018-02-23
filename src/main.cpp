#include <Arduino.h>
#include <NeuralNetwork.h>


#define LEARNINGRATE  0.15
#define MOMENTUM      0.5
#define TOPOLOGY     {3, 2, 1}


NeuralNetwork  net(TOPOLOGY);
vector<double> input, target, result;


void setup() {

  net.begin(LEARNINGRATE, MOMENTUM);
}

void loop() {

  // TODO: training function

  net.feedForward(input);
  net.propBack(target);

  result = net.getOutput();
}
