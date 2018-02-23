#include <Arduino.h>
#include <NeuralNetwork.h>
#include <TrainingData.h>


#define TOPOLOGY     {16, 32, 8, 2}
#define LEARNINGRATE  0.15
#define MOMENTUM      0.5


NeuralNetwork net(TOPOLOGY);


Table line = {
  -1, -1, -1, -1,
  -1, -1, -1,  1,
  -1, -1,  1, -1,
  -1,  1, -1, -1
};

Table circle = {
  -1,  1,  1, -1,
   1, -1, -1,  1,
   1, -1, -1,  1,
  -1,  1,  1, -1
};


void setup() {

  Serial.begin(9600);
  Serial.println(" ");

  net.begin(LEARNINGRATE, MOMENTUM);
  net.train(trainingData_img, 1000);
}

void loop() {

  Serial.println("- - - - - - -");

  net.feedForward(circle);
  Table output = net.getOutput();
  Serial.print(output[0]); Serial.print(" | "); Serial.println(output[1]);

  net.feedForward(line);
  output = net.getOutput();
  Serial.print(output[0]); Serial.print(" | "); Serial.println(output[1]);

  delay(2000);
}
