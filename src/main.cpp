#include <Arduino.h>

#include <NeuralNetwork.h>
  #include <Exercises.h>
  #include <Memories.h>


#define TOPOLOGY     {16, 4, 4, 1}
#define LEARNINGRATE  0.15
#define MOMENTUM      0.5


NeuralNetwork net(TOPOLOGY);


List testInput = {
  0, 1, 0, 0,
  0, 1, 0, 0,
  0, 1, 0, 0,
  0, 1, 0, 0
};


void setup() {

  Serial.begin(9600); Serial.println(); Serial.println();

  net.begin(LEARNINGRATE, MOMENTUM);
  net.train(exercise_img, 10000).memorize();
  // net.recall(memory_img_0727_20180228).feedForward(testInput);

  // List output = net.getOutput();
  // Serial.println(output[0]);
}

void loop() {


}
