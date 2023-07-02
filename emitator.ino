#include <Arduino.h>
#include <cstring>
#include <Wire.h>
#include <WiFi.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <esp_now.h>


Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

esp_now_peer_info_t peerInfo;

void sendData(uint8_t* data, size_t sizeBuff) {
  esp_now_send(peerInfo.peer_addr, data, sizeBuff);
}


void setup(void)
{
  Serial.begin(921600);
  delay(1000);
  Serial.println("Accelerometer Test");
  Serial.println("");
  Wire.setClock(400000);
  
  if (!accel.begin())
  {
    Serial.println("Ooops, no ADXL345 detected ... Check your wiring!");
  }
  Wire.setClock(400000);
  //setarea frecvenței de eșantionare
  accel.setDataRate(ADXL345_DATARATE_800_HZ); 
  
  //setarea sensibilității
  //accel.setRange(ADXL345_RANGE_16_G)
  accel.setRange(ADXL345_RANGE_8_G);
  //accel.setRange(ADXL345_RANGE_4_G);
  //accel.setRange(ADXL345_RANGE_2_G);
   
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  //memset(destination, value, N_bytes);
  memset(&peerInfo, 0, sizeof(peerInfo));
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  memcpy(peerInfo.peer_addr, "\xFF\xFF\xFF\xFF\xFF\xFF", ESP_NOW_ETH_ALEN);
  esp_now_add_peer(&peerInfo);
  
  Serial.println("");
}

unsigned long startTime, stopTime, deltaTime;

void loop(void)
{
  sensors_event_t event;
  
  String pachetToSend = "";
  String dataToSend;
  stopTime = micros();
  int i = 0;
  while(i < 10) {
    startTime = micros();
    deltaTime = startTime - stopTime;
    if ((float)deltaTime <= 2000 && (float)deltaTime >= 1500){
      Serial.println(deltaTime);
      stopTime = micros();
      accel.getEvent(&event);
      dataToSend = String(event.acceleration.x) + ";" + String(event.acceleration.y) + ";" + String(event.acceleration.z) + ";";
      pachetToSend = pachetToSend + dataToSend;
      i++;
    }
  }
  uint8_t* buffer = (uint8_t*)pachetToSend.c_str();
  size_t sizeBuff =  pachetToSend.length();
  sendData(buffer, sizeBuff);
}
