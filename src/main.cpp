#include <Arduino.h>
#include <NeuralNetwork.h>


#define LEARNINGRATE  0.15
#define MOMENTUM      0.0
#define TOPOLOGY     {2, 4, 1}


NeuralNetwork  net(TOPOLOGY);

uint32_t pass = 0;

void setup() {

  Serial.begin(9600);
  net.begin(LEARNINGRATE, MOMENTUM);

  for (size_t i = 0; i < 10000; i++) {

    vector<double> input, target, output;
    uint8_t num = random(0, 3);

    switch (num) {
      case 0: input = {0, 0}; target = {0}; break;
      case 1: input = {0, 1}; target = {1}; break;
      case 2: input = {1, 0}; target = {1}; break;
      case 3: input = {1, 1}; target = {0}; break;
    }

    net.feedForward(input);
    net.propBack(target);

    pass++;

    yield();
  }
}

void loop() {

  // TODO: training function

  vector<double> input, target, output;
  uint8_t num = random(0, 3);

  switch (num) {
    case 0: input = {0, 0}; target = {0}; break;
    case 1: input = {0, 1}; target = {1}; break;
    case 2: input = {1, 0}; target = {1}; break;
    case 3: input = {1, 1}; target = {0}; break;
  }

  net.feedForward(input);

  output = net.getOutput();

  Serial.print("- - - - "); Serial.print("Pass "); Serial.print(pass); Serial.println(" - - - -");
  Serial.print("Input  = { "); Serial.print(input[0]); Serial.print(" | "); Serial.print(input[1]); Serial.println(" }");
  Serial.print("Target =   "); Serial.println(target[0]);
  Serial.print("Actual =   "); Serial.println(output[0]);
  Serial.println("- - - - - - - - - - - - -"); Serial.println();

  net.propBack(target);

  delay(2000);

  pass++;
}
