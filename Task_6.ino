
/*
* Filename: Task_6.ino
* Functions: forward(), turnLeft(), turnRight(), stopMotors(), uTurn(), 
* Global Variables: ssid, password, port, host, leftSensorPin1, rightSensorPin1, leftSensorPin2, rightSensorPin2, leftMotorForward, leftMotorBackward, 
* rightMotorForward, rightMotorBackward, l1, l2, r1, r2, buzzer, msg, i, got, final_end, incomingPacket, st, counter
 */


#include <WiFi.h>

// WiFi credentials
const char* ssid = "akash";                    //Enter your wifi hotspot ssid
const char* password =  "akashsingh";          //Enter your wifi hotspot password
const uint16_t port = 8002;                     // Port number for communication
const char * host = "192.168.47.126";           // Host IP address

// Pin configurations for sensors

const int leftSensorPin1 = 22;                 // Pin for left sensor 1
const int rightSensorPin1 = 23;                // Pin for right sensor 1
const int leftSensorPin2 = 19;                 // Pin for left sensor 2
const int rightSensorPin2 = 21;                // Pin for right sensor 2

// Pin configurations for motors
const int leftMotorForward = 27;               // Pin for left motor forward
const int leftMotorBackward = 4;               // Pin for left motor backward
const int rightMotorForward = 2;               // Pin for right motor forward
const int rightMotorBackward = 15;             // Pin for right motor backward

// Variables for sensor readings
int l1=0;                                     // Flag for left sensor 1
int l2=0;                                     // Flag for left sensor 2
int r1=0;                                     // Flag for right sensor 1
int r2=0;                                     // Flag for right sensor 2

// Pin configuration for the buzzer
int buzzer = 5;                               // Pin for the buzzer

// Message string for communication
String msg="0";                               // Message string

// Counter variable
int i=10;                                     // Counter variable  

// Flag variables
int got=0;                                    // Flag for message received
int end=0;                                    // Flag for end condition
int final_end=0;                              // Final end flag


// Buffer for incoming packets
char incomingPacket[80];                      // Buffer to store incoming packets

// WiFi client object
WiFiClient client;                            // WiFi client object

// String variable
String st="0";                                // String variable

// A variable to read the next move 
int counter = 0;

void setup() {
  /*
 * Function Name: setup
 * Input: None
 * Output: None
 * Logic: This function initializes the serial communication, sets up pin modes for sensors,
 *        motors, and buzzer, connects to the WiFi network using the provided credentials,
 *        and establishes a connection to the specified host and port. It also reads a message
 *        from the host, sends an acknowledgment, and initializes the counter variable.
 * ExampleCall: setup()
 */

  // Initialize serial communication at a baud rate of 115200 bits per second
  Serial.begin(115200);                     
  
  // Set pin modes for sensors
  pinMode(leftSensorPin1, INPUT);
  pinMode(rightSensorPin1, INPUT);
  pinMode(leftSensorPin2, INPUT);
  pinMode(rightSensorPin2, INPUT);

  // Set pin modes for motors
  pinMode(leftMotorForward, OUTPUT);
  pinMode(leftMotorBackward, OUTPUT);
  pinMode(rightMotorForward, OUTPUT); 
  pinMode(rightMotorBackward, OUTPUT); 

  // Set pin mode for the buzzer
  pinMode(buzzer,OUTPUT);
  digitalWrite(buzzer,HIGH);                    // Set buzzer initially to HIGH

  // Connect to WiFi network
  WiFi.begin(ssid, password);
 
  // Wait until connected to WiFi
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }
 
  // Print local IP address once connected
  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());
 
  // Attempt to establish connection to host and port
  while (!client.connect(host, port))
  {
     Serial.println("Connection to host failed");
  }
     // Read message from host until newline character
      msg = client.readStringUntil('\n');                    //Read the message through the socket until new line char(\n)

      // Send acknowledgment to host
      client.print("Hello from ESP32!");                     //Send an acknowledgement to host

      // Convert received message to integer and assign to counter variable
      counter = msg.toInt();
      Serial.println(counter);                               // Print counter value
}

void loop() {

  /*
 * Function Name: loop
 * Input: None
 * Output: None
 * Logic: This function continuously runs in a loop, handling communication with the host
 *        and executing commands based on received messages. It sends acknowledgments to
 *        the host and performs movements such as forward, backward, left turn, right turn,
 *        and U-turn based on the received commands. It also handles obstacle detection and
 *        adjusts the vehicle's movement accordingly. Additionally, it handles special commands
 *        like stopping at nodes and final stops. It utilizes various sensor readings to navigate
 *        and communicates with the host for updates and acknowledgments.
 * ExampleCall: the compiler loops this set of program 
 */

  digitalWrite(buzzer,HIGH);                                   // Activate buzzer

  client.print(" red ");                                        //Send an read acknowledgement to host 

  // Read message from host until newline character
  msg = client.readStringUntil('\n'); 
  
  // Reset message if final end flag is set
  if(final_end==1){
    msg=""; 

  }

  // Get length of received message
  int n=msg.length();

  // Send message length to host
  client.print(n);

  // Process message if message length is greater than 0
  if(n>0){
      i=0;                                                   // Reset counter variable
      char counter=msg[i];                                   // Get first character of the message


      // Execute corresponding action based on the received command      
      if(counter=='r')
      {                               
        turnRight();                                          //right turn 
        delay(500);                                           // Wait until either left or right sensor detects an obstacle

        // Flags defind same as above
        int l1=0;                                           
        int l2=0;

        // for proper turning to right 
        while(l1+l2!=1)
        {                                     
          if(digitalRead(leftSensorPin1) == HIGH)
           {
             l1=1;
           }
          if(digitalRead(rightSensorPin1) == HIGH)
           {
            l2=1;
           }
        }
      }
      else if(counter=='f')                            //  will move forward when value received from the host is 'f'
      {

       // Forward
       forward();

        if(n>=i)
        {
          if (digitalRead(leftSensorPin1) == HIGH && digitalRead(rightSensorPin1) == HIGH)
            {
               delay(170);
            }
        }
          
      }
      else if(counter=='l')                              //  will move left when value received from the host is 'l'      
      {
         turnLeft();
         delay(500);
         int l1=0;
         int l2=0;
         while(l1+l2!=1)
          {
          if(digitalRead(leftSensorPin1) == HIGH)
            {
            l1=1;
            }
          if(digitalRead(rightSensorPin1) == HIGH)
            {
            l2=1;
            }
          }
      }
       else if(counter=='u')                                //  will move uTurn when value received from the host is 'u'  
       {
        uTurn();
       }
        i++;                                               // Increment counter variable 
       if(i==n)
        {
         client.print(" ln ");                                // Send "ln" to host if all characters in message are processed
        }
      }

  // Continue processing if there are still characters in the message
  while(n>=i){
     // Read sensor values
    l1=digitalRead(leftSensorPin1);
    l2=digitalRead(leftSensorPin2);
    r1=digitalRead(rightSensorPin1);
    r2=digitalRead(rightSensorPin2);

    // Check for node condition (both left and right sensors detect obstacle)
    if(l1==1 && r1==1){ 

      // Send "Node" to host
      client.print(" Node ");

      stopMotors();                                        // Stop motors
      char counter=msg[i];

      // Execute corresponding action based on the next character in the message
      if(counter=='r'){
       turnRight();
       delay(400);

        // Wait until either left or right sensor detects an obstacle
       int l1=0;
       int l2=0;

       while(l1+l2!=1){
        if(digitalRead(leftSensorPin1) == HIGH){
         l1=1;
        }
        if(digitalRead(rightSensorPin1) == HIGH){
         l2=1;
        }
      }
    } else if(counter=='l'){ 
      turnLeft();
      delay(400);
      // Wait until either left or right sensor detects an obstacle
      int l1=0;
      int l2=0;

      while(l1+l2!=1){
        if(digitalRead(leftSensorPin1) == HIGH){
          l1=1;
        }
        if(digitalRead(rightSensorPin1) == HIGH){
          l2=1;
        }
       }
      } else if(counter=='f'){
       
        forward();                                  // forward

        // Adjust delay based on message length
        if(n>i){
          delay(170);
        }
        else{
            delay(160);
          }  
        }else if(counter=='u'){
          uTurn();
        }
        i++;                                             // Increment counter variable
        forward();

      // Send "ln" to host if all characters in message are processed
        if(i==n){
         client.print(" ln ");
        }
    }
    else if(l2==1 || l1==1){ //left
      turnLeft();
    }
    else if(r2==1 || r1==1){ //right
      turnRight();
    }
    else{                   
      forward();
    }
    // Check if all characters in the message are processed
    if(i==n){


      client.setTimeout(0.1);                    // Read special commands from the host
    st=client.readStringUntil('\n');
    client.print(st);                             // Send response to host
    if(st==" end "){
      
      // Stop motors and buzzer if "end" command is received
        stopMotors();
        digitalWrite(buzzer,LOW);
        client.print(" final "); 
        delay(5000);
        msg="";
        final_end=1;
        break;
    }
     // Stop motors and buzzer if "stop" command is received
    if(got==1){
     client.print(" got=1  ");
    stopMotors();
    digitalWrite(buzzer,LOW);
    delay(1000);
    got=0;
    i++;
    break;
    }
    if(st==" stop "){
    
      client.print(" ld ");
      got=1;
    }
  }
  
}

  client.print(" done ");          //Send an acknowledgement to host
  stopMotors();
  got=0;                          // Reset flag variable
}

void forward() {

/*
 * Function Name: forward
 * Input: None
 * Output: None
 * Logic: This function sets the left and right motors to move forward by
 *        setting the appropriate digital pins to HIGH for forward motion
 * ExampleCall: forward()
 */


  digitalWrite(leftMotorForward, HIGH);    // Set left motor forward pin to HIGH
  digitalWrite(leftMotorBackward, LOW);    // Set left motor backward pin to LOW
  digitalWrite(rightMotorForward, HIGH);   // Set right motor forward pin to HIGH
  digitalWrite(rightMotorBackward, LOW);   // Set right motor backward pin to LOW 
}

void turnLeft() {

  /*
 * Function Name: turnLeft
 * Input: None
 * Output: None
 * Logic: This function turns the vehicle to the left by stopping the left motor
 *        and moving the right motor forward, causing the vehicle to pivot left.
 * ExampleCall: turnLeft()
 */

  digitalWrite(leftMotorForward, LOW);   // Stop left motor forward motion
  digitalWrite(leftMotorBackward, LOW);   // Stop left motor backward motion
  
  digitalWrite(rightMotorForward, HIGH);  // Set right motor forward pin to HIGH
  digitalWrite(rightMotorBackward, LOW);  // Set right motor backward pin to LOW

}



void turnRight() {
  /*
 * Function Name: turnRight
 * Input: None
 * Output: None
 * Logic: This function turns the vehicle to the right by stopping the right motor
 *        and moving the left motor forward, causing the vehicle to pivot right.
 * ExampleCall: turnRight()
 */
 
  digitalWrite(leftMotorForward, HIGH);   // Set left motor forward pin to HIGH
  digitalWrite(leftMotorBackward, LOW);   // Set left motor backward pin to LOW
  
  digitalWrite(rightMotorForward, LOW);   // Stop right motor forward motion
  digitalWrite(rightMotorBackward, LOW);  // Stop right motor backward motion
}


void stopMotors() {

  /*
 * Function Name: stopMotors
 * Input: None
 * Output: None
 * Logic: This function stops both the left and right motors of the vehicle by 
 *        setting their respective forward and backward pins to LOW.
 * ExampleCall: stopMotors()
 */
 
  digitalWrite(leftMotorForward, LOW);    // Stop left motor forward motion
  digitalWrite(leftMotorBackward, LOW);   // Stop left motor backward motion
  
  digitalWrite(rightMotorForward, LOW);   // Stop right motor forward motion
  digitalWrite(rightMotorBackward, LOW);  // Stop right motor backward motion
}

void uTurn() {
/*
 * Function Name: uTurn
 * Input: None
 * Output: None
 * Logic: This function performs a U-turn by moving the left motor forward and the right motor backward
 *        for a specified duration. It then waits until either the left or right sensor detects an obstacle.
 * ExampleCall: uTurn()
 */

  digitalWrite(leftMotorForward, HIGH);    // Set left motor forward pin to HIGH
  digitalWrite(leftMotorBackward, LOW);    // Set left motor backward pin to LOW
  
  digitalWrite(rightMotorForward, LOW);    // Set right motor forward pin to LOW
  digitalWrite(rightMotorBackward, HIGH);  // Set right motor backward pin to HIGH
  
  delay(800);  // Allow the vehicle to perform the turn
  
  int l1 = 0;  // Variable to track left sensor detection
  int l2 = 0;  // Variable to track right sensor detection
  
  // Wait until either the left or right sensor detects an obstacle
  while (l1 + l2 != 1) {
    if (digitalRead(leftSensorPin2) == HIGH) {
      l1 = 1;  // Set l1 to 1 if left sensor detects an obstacle
    }
    if (digitalRead(rightSensorPin2) == HIGH) {
      l2 = 1;  // Set l2 to 1 if right sensor detects an obstacle
    }
  }
}