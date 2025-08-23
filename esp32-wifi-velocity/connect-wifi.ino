#include <WiFi.h>
#include "credentials.h"  // loads ssid and password
#include "velocityConversions.h"

int max_count = 1000;
int port = 8888;

WiFiUDP UDP;
IPAddress ip(192, 168, 8, 10);

#define TX 17
#define RX 16
HardwareSerial robotSerial(2);

void setup() {
  Serial.begin(115200);
  robotSerial.begin(115200, SERIAL_8N1, RX, TX);
  connect_wifi();
  UDP.begin(port);
}

void loop() {
  int size = UDP.parsePacket();
  if (size) {
    char buffer[512];  // Larger buffer for multi-line commands
    Serial.printf("Packet size: %d bytes from %s:%d\n", size, UDP.remoteIP().toString().c_str(), UDP.remotePort());
    size = UDP.read(buffer, sizeof(buffer) - 1);
    buffer[size] = '\0';
    Serial.printf("Received: %s\n", buffer);
    UDP.flush();

    // Process the received data (handle both single and multi-line formats)
    processReceivedData(buffer, size);
  }
}

void processReceivedData(char* buffer, int size) {
  String data = String(buffer);
  data.trim();

  // Check if it's multi-line format (from TritonClient actions)
  if (data.indexOf('\n') >= 0 || data.endsWith("\0")) {
    Serial.println("Processing multi-line format");
    processMultiLineCommands(data);
  } else {
    Serial.println("Processing single command");
    processSingleCommand(data);
  }
}

void processMultiLineCommands(String data) {
  // Remove null terminator if present
  data.replace("\0", "");

  // Split by newlines and process each command
  int start = 0;
  int end = data.indexOf('\n');

  while (end >= 0) {
    String command = data.substring(start, end);
    command.trim();

    if (command.length() > 0) {
      Serial.printf("Processing line: '%s'\n", command.c_str());
      processSingleCommand(command);
    }

    start = end + 1;
    end = data.indexOf('\n', start);
  }

  // Process remaining part after last newline
  String lastCommand = data.substring(start);
  lastCommand.trim();
  if (lastCommand.length() > 0) {
    Serial.printf("Processing final line: '%s'\n", lastCommand.c_str());
    processSingleCommand(lastCommand);
  }
}

void processSingleCommand(String command) {
  // Parse the command and extract movement parameters
  double vx = 0.0, vy = 0.0, rotV = 0.0;
  bool dribble_on = false;

  bool valid_command = false;

  if (command.startsWith("kick")) {
    // Parse "kick vx vy" command
    int space1 = command.indexOf(' ');
    int space2 = command.indexOf(' ', space1 + 1);
    if (space1 > 0 && space2 > 0) {
      vx = command.substring(space1 + 1, space2).toDouble() * 0.5;  // Moderate scaling
      vy = command.substring(space2 + 1).toDouble() * 0.5;
      dribble_on = true;
      valid_command = true;
    }
  } else if (command.startsWith("dash")) {
    // Parse "dash speed" command
    int space1 = command.indexOf(' ');
    if (space1 > 0) {
      vy = command.substring(space1 + 1).toDouble() * 0.5;  // Forward movement
      dribble_on = false;
      valid_command = true;
    }
  } else if (command.startsWith("turn")) {
    // Parse "turn speed" command
    int space1 = command.indexOf(' ');
    if (space1 > 0) {
      rotV = command.substring(space1 + 1).toDouble() * 0.1;  // Smaller rotation scaling
      dribble_on = false;
      valid_command = true;
    }
  } else if (command == "bye" || command == "stop") {
    // Stop command
    vx = vy = rotV = 0.0;
    dribble_on = false;
    valid_command = true;
  } else if (command.startsWith("test")) {
    // Test command with small known values
    vx = 0.0;
    vy = 10.0;  // Small forward movement
    rotV = 0.0;
    dribble_on = false;
    valid_command = true;
  } else {
    Serial.printf("Unknown command: '%s'\n", command.c_str());
    return;
  }

  if (valid_command) {
    Serial.printf("Parsed: vx=%.3f, vy=%.3f, rotV=%.3f, dribble=%d\n", vx, vy, rotV, dribble_on);
    sendMotorCommands(vx, vy, rotV, dribble_on);
  }
}

void sendMotorCommands(double vx, double vy, double rotV, bool dribble_on) {
  std::array<uint8_t, 8> msg;
  action_to_byte_array_with_params(msg, vx, vy, rotV);

  // Debug: show the motor data bytes
  Serial.print("Motor bytes: ");
  for (int i = 0; i < 8; i++) {
    Serial.printf("0x%02X ", msg[i]);
  }
  Serial.println();

  std::array<uint8_t, 2> header = {0xca, 0xfe};
  std::array<uint8_t, 1> dribble = {0x00};
  dribble[0] = dribble_on ? 0x01 : 0x00;

  // Send header bytes with small delays to ensure STM32 can process sequentially
  robotSerial.write(header[0]);
  delayMicroseconds(100);  // Small delay for STM32 interrupt processing
  robotSerial.write(header[1]);
  delayMicroseconds(100);  // Small delay for STM32 interrupt processing

  // Send data bytes
  robotSerial.write(msg.data(), msg.size());
  delayMicroseconds(50);   // Small delay before dribble byte

  // Send dribble byte
  robotSerial.write(dribble[0]);

  // Debug output
  Serial.print("Sent to STM32: ");
  for (int i = 0; i < 2; i++) {
    Serial.printf("0x%02X ", header[i]);
  }
  for (int i = 0; i < 8; i++) {
    Serial.printf("0x%02X ", msg[i]);
  }
  Serial.printf("0x%02X\n", dribble[0]);
}

void connect_wifi() {
  Serial.print("\nConnecting WiFi to ");
  Serial.print(ssid);
  // Attempt connection every 500 ms
  WiFi.config(ip);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.print("\nWiFi connected");
  Serial.print("\nIP address: ");
  Serial.println(WiFi.localIP());
}
