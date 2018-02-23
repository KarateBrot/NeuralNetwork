#include <Arduino.h>
#include <NeuralNetwork.h>
#include <TrainingData.h>


#define TOPOLOGY     {2, 4, 1}
#define LEARNINGRATE  0.15
#define MOMENTUM      0.3


NeuralNetwork net(TOPOLOGY);

uint32_t pass = 0;


void setup() {

  Serial.begin(9600);
  Serial.println(" ");

  net.begin(LEARNINGRATE, MOMENTUM);

  // Train network to become xor gate
  net.train(trainingData_xor, 2000);
}

void loop() {

  vector<double> input, target, output;
    uint8_t num = random(0, 4);

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
    Serial.print("AvgErr =   "); Serial.println(net.getAvgError(), 5);
    Serial.println("- - - - - - - - - - - - -"); Serial.println();

    net.propBack(target);

    delay(2000);

  pass++;
}
