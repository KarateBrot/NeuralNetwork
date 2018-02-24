#include <Arduino.h>

#include <NeuralNetwork.h>
  #include <Exercises.h>
  #include <Memories.h>


#define TOPOLOGY     {2, 4, 1}
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
  net.recall(memory_xor_0404_20180224);
  net.train(exercise_xor, 10000);
}

void loop() {

  net.memorize();

  delay(2000);
}
