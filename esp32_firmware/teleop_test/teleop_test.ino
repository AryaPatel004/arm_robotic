#include <Preferences.h>
#include <esp_arduino_version.h>

const int SERVO_PINS[4] = {5, 16, 17, 18};
const char *NAMES[4] = {"base", "shoulder", "elbow", "gripper"};
const char *KEYS[4] = {"a0", "a1", "a2", "a3"};

const int PIN_J1_X  = 32;
const int PIN_J1_Y  = 33;
const int PIN_J1_SW = 34;
const int PIN_J2_X  = 35;
const int PIN_J2_Y  = 36;
const int PIN_J2_SW = 39;

const int ANGLE_MIN = 0;
const int ANGLE_MAX = 180;
const float HOME_POSE[4] = {85.7, 26.4, 72.1, 81.0};
const int PULSE_MIN_US = 500;
const int PULSE_MAX_US = 2400;

const int PWM_FREQ = 50;
const int PWM_BITS = 16;
const float PERIOD_US = 20000.0;

const int DEADZONE = 150;
const float DEG_PER_SEC_AT_FULL_DEFLECTION = 30.0;
const float HOMING_DEG_PER_SEC = 5.0;
const float MAX_DEG_PER_SEC = 30.0;
const unsigned long SAVE_INTERVAL_MS = 1000;

Preferences prefs;
float cmd[4];
float tgt[4];
float saved[4];
bool homing = true;

unsigned long lastUpdateMs = 0;
unsigned long lastPrintMs = 0;
unsigned long lastSaveMs = 0;

int centerJ1x, centerJ1y, centerJ2x, centerJ2y;

float angleToUs(float a) {
  return PULSE_MIN_US + (a - ANGLE_MIN) * (PULSE_MAX_US - PULSE_MIN_US) / (float)(ANGLE_MAX - ANGLE_MIN);
}

uint32_t usToDuty(float us) {
  return (uint32_t)round(us * ((1UL << PWM_BITS) - 1) / PERIOD_US);
}

void servoAttach(int i, float angle) {
  uint32_t duty = usToDuty(angleToUs(angle));
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(SERVO_PINS[i], PWM_FREQ, PWM_BITS);
  ledcWrite(SERVO_PINS[i], duty);
#else
  ledcSetup(i, PWM_FREQ, PWM_BITS);
  ledcWrite(i, duty);
  ledcAttachPin(SERVO_PINS[i], i);
#endif
}

void servoWrite(int i, float angle) {
  uint32_t duty = usToDuty(angleToUs(angle));
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(SERVO_PINS[i], duty);
#else
  ledcWrite(i, duty);
#endif
}

int calibrateCenter(int pin) {
  long sum = 0;
  for (int i = 0; i < 50; i++) { sum += analogRead(pin); delay(2); }
  return sum / 50;
}

float applyDeadzone(int raw, int center) {
  int delta = raw - center;
  if (abs(delta) < DEADZONE) return 0.0;
  int sign = delta > 0 ? 1 : -1;
  int magnitude = abs(delta) - DEADZONE;
  int maxMagnitude = (delta > 0 ? 4095 - center : center) - DEADZONE;
  if (maxMagnitude <= 0) return 0.0;
  return constrain(sign * (float)magnitude / maxMagnitude, -1.0, 1.0);
}

void warnIfAtLimit(int i) {
  if (cmd[i] <= ANGLE_MIN || cmd[i] >= ANGLE_MAX) {
    Serial.printf("WARNING: %s at its end stop (%d deg) - back off if it buzzes or gets warm\n", NAMES[i], (int)cmd[i]);
  }
}

void savePosition(unsigned long now) {
  if (now - lastSaveMs < SAVE_INTERVAL_MS) return;
  bool changed = false;
  for (int i = 0; i < 4; i++) {
    if (fabs(cmd[i] - saved[i]) >= 1.0) changed = true;
  }
  if (!changed) return;
  lastSaveMs = now;
  for (int i = 0; i < 4; i++) {
    saved[i] = cmd[i];
    prefs.putFloat(KEYS[i], saved[i]);
  }
}

void setup() {
  prefs.begin("arm", false);
  bool known = prefs.isKey(KEYS[0]);
  for (int i = 0; i < 4; i++) {
    saved[i] = constrain(prefs.getFloat(KEYS[i], HOME_POSE[i]), (float)ANGLE_MIN, (float)ANGLE_MAX);
    cmd[i] = saved[i];
    tgt[i] = HOME_POSE[i];
    servoAttach(i, cmd[i]);
  }

  Serial.begin(115200);
  pinMode(PIN_J1_SW, INPUT);
  pinMode(PIN_J2_SW, INPUT);

  if (!known) {
    Serial.println("first boot - position unknown, place the arm near the 0th pose by hand");
  }
  Serial.printf("holding last pose: base=%.1f shoulder=%.1f elbow=%.1f gripper=%.1f\n", cmd[0], cmd[1], cmd[2], cmd[3]);

  Serial.println("Calibrating joystick centers - do not touch the sticks...");
  centerJ1x = calibrateCenter(PIN_J1_X);
  centerJ1y = calibrateCenter(PIN_J1_Y);
  centerJ2x = calibrateCenter(PIN_J2_X);
  centerJ2y = calibrateCenter(PIN_J2_Y);
  Serial.printf("Centers: J1x=%d J1y=%d J2x=%d J2y=%d\n", centerJ1x, centerJ1y, centerJ2x, centerJ2y);

  Serial.printf("homing slowly to 0th pose at %.1f deg/s - sticks ignored until done\n", HOMING_DEG_PER_SEC);
  lastUpdateMs = millis();
  lastSaveMs = lastUpdateMs;
}

void loop() {
  unsigned long now = millis();
  float dtSec = (now - lastUpdateMs) / 1000.0;
  lastUpdateMs = now;

  int j1x = analogRead(PIN_J1_X);
  int j1y = analogRead(PIN_J1_Y);
  int j1sw = digitalRead(PIN_J1_SW);
  int j2x = analogRead(PIN_J2_X);
  int j2y = analogRead(PIN_J2_Y);
  int j2sw = digitalRead(PIN_J2_SW);

  bool frozen = (j1sw == LOW);

  if (!homing && !frozen) {
    tgt[0] += applyDeadzone(j1x, centerJ1x) * DEG_PER_SEC_AT_FULL_DEFLECTION * dtSec;
    tgt[1] += applyDeadzone(j1y, centerJ1y) * DEG_PER_SEC_AT_FULL_DEFLECTION * dtSec;
    tgt[2] += applyDeadzone(j2x, centerJ2x) * DEG_PER_SEC_AT_FULL_DEFLECTION * dtSec;
    tgt[3] = map(j2y, 0, 4095, ANGLE_MIN, ANGLE_MAX);
  }

  float maxStep = (homing ? HOMING_DEG_PER_SEC : MAX_DEG_PER_SEC) * dtSec;
  bool allThere = true;
  for (int i = 0; i < 4; i++) {
    tgt[i] = constrain(tgt[i], (float)ANGLE_MIN, (float)ANGLE_MAX);
    float d = tgt[i] - cmd[i];
    if (fabs(d) > 0.1) allThere = false;
    cmd[i] += constrain(d, -maxStep, maxStep);
    servoWrite(i, cmd[i]);
  }

  if (homing && allThere) {
    homing = false;
    Serial.println("at 0th pose - joystick control ON");
  }

  savePosition(now);

  if (now - lastPrintMs > 200) {
    lastPrintMs = now;
    Serial.printf("J1 x=%d y=%d sw=%d | J2 x=%d y=%d sw=%d | base=%.1f shoulder=%.1f elbow=%.1f gripper=%.1f%s%s\n",
                  j1x, j1y, j1sw, j2x, j2y, j2sw, cmd[0], cmd[1], cmd[2], cmd[3],
                  homing ? " [HOMING]" : "", frozen ? " [FROZEN]" : "");
    for (int i = 0; i < 3; i++) warnIfAtLimit(i);
  }
}
