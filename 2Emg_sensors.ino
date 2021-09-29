#include <Servo.h>
Servo idxMidFinger;
Servo ringPinkyFinger;

bool pinkyActive = false;
bool idxActive = false;
//the end number of 1 means that it is associated with the index and the middle finger
//the end number of 2 means that it is associated with the ring and pinky finger

int emg1 = 0;
int emg2 = 1;

//bool inMotion = false; //will determine if the finger is in motion based on the value returned from the emg sensor

int servoPosition1 = 1; //this value will be assigned to the servo motor
int servoPosition2 = 1; //this value will be assigned to the second servo motor

int valuesFromEmg1[5]; //should hold values from the sensor to get an average value
//This is because sometimes there are extreme inaccuracies that can throw off the code so there needs to be a way to get an average 

int valuesFromEmg2[5];

int counter = 0; //this is used to keep track of the index to insert the value at
//you don't want to replace values that you just put in

void setup() 
{
  idxMidFinger.attach(5);
  ringPinkyFinger.attach(6);
  Serial.begin(9600); //this will be used to visualize and monitor the values coming from the sensor
}

void loop()
{
  //General plan:
//  1.) get a value from the emg sensor
//  - most of the values are at around 200 when you contract
//  - therefore, cap the values at 200
//  2.) take the average reading from the past 5 emg readings
//  3.) if the value is 200 or greater, cap at 200 and then actuate the servo motor by setting it to the degree at contraction
//  4.) once the average value is no longer 200, then you need to release the motor and set it back to the degree at resting position

  //1.) receive a value from the emg sensor
  int emgVal1 = analogRead(emg1); //read the value coming from the signal pin attached to the analog pin of the EMG sensor
  int emgVal2 = analogRead(emg2);
  
  int avgEmgVal1 = averageValues(emgVal1, valuesFromEmg1);
  int avgEmgVal2 = averageValues(emgVal2, valuesFromEmg2);
  
  //2.) capping emgVal at 200                                             
  if (avgEmgVal1 >= 280) //threshold set to 200 because only the middle finger at spot 
  {
    avgEmgVal1 = 300;
//    inMotion = true;
  }
//  else if (avgEmgVal < 200)
//  {
//    inMotion = false;
//  }


  if (avgEmgVal2 >= 320) //threshold set to 200 because only the middle finger at spot 
  {
    avgEmgVal2 = 400;
//    inMotion = true;
  }
//  else if (avgEmgVal2 < 200)
//  {
//    inMotion = false;
//  }

  if (avgEmgVal1 == 300) //index
  {
    idxMidFinger.write(25);
    idxActive = true;
  }
  else
  {
    idxMidFinger.write(180);
    idxActive = false;
  }

  if (avgEmgVal2 == 400) //pimky
  {
    ringPinkyFinger.write(160);
    pinkyActive = true;
  }
  else
  {
    ringPinkyFinger.write(25);
    pinkyActive = false;
  }
  
  
  //Serial.print("Current reading: ");
  Serial.print(emgVal1);
  Serial.print(" ");
  Serial.println(emgVal2);
  delay(50); 
}

int averageValues(int emgVal, int* valuesFromEmg)
{
  if (counter == 5)
  {
    counter = 0;
  }
  valuesFromEmg[counter] = emgVal;
  counter++;

  int total = 0;
  for (int i = 0; i < 5; i++)
  {
    total += valuesFromEmg[i];
  }
  //Serial.println("\n");
  return (int) (total / 5);
}
