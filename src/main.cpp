#include <Arduino.h>

#include <NeuralNetwork.h>
  #include <Exercises.h>
  #include <Memories.h>


#define TOPOLOGY     {16, 4, 1}
#define LEARNINGRATE  0.15
#define MOMENTUM      0.5


NeuralNetwork net(TOPOLOGY);


Table testInput = {
  0, 1, 0, 0,
  0, 1, 0, 0,
  0, 1, 0, 0,
  0, 1, 0, 0
};


void setup() {

  Serial.begin(9600); Serial.println(); Serial.println();

  net.begin(LEARNINGRATE, MOMENTUM);

  net.train(exercise_img, 10000);
  net.memorize();

  // net.recall(memory_img2_0033_20180225);
}

void loop() {


}
