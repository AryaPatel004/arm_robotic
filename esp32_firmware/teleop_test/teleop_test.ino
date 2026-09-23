#include <ESP32Servo.h>
#include <Preferences.h>

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
const int START_ANGLE = 90;
const int PULSE_MIN_US = 500;
const int PULSE_MAX_US = 2400;

const int DEADZONE = 150;
// const float DEG_PER_SEC_AT_FULL_DEFLECTION = 90.0;
// const float STARTUP_DEG_PER_SEC = 20.0;
// const float MAX_DEG_PER_SEC = 90.0;
const float DEG_PER_SEC_AT_FULL_DEFLECTION = 30.0;
const float STARTUP_DEG_PER_SEC = 10.0;
const float MAX_DEG_PER_SEC = 30.0;
const unsigned long SAVE_AFTER_STILL_MS = 1000;

Servo servos[4];
Preferences prefs;
float cmd[4];
float tgt[4];
float saved[4];
bool starting = true;
bool hadSaved = false;

unsigned long lastUpdateMs = 0;
unsigned long lastPrintMs = 0;
unsigned long lastMoveMs = 0;

int centerJ1x, centerJ1y, centerJ2x, centerJ2y;

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

void saveIfSettled(unsigned long now) {
  if (starting || now - lastMoveMs < SAVE_AFTER_STILL_MS) return;
  bool changed = false;
  for (int i = 0; i < 4; i++) {
    if (fabs(cmd[i] - saved[i]) >= 1.0) changed = true;
  }
  if (!changed) return;
  for (int i = 0; i < 4; i++) {
    saved[i] = cmd[i];
    prefs.putFloat(KEYS[i], saved[i]);
  }
  Serial.printf("saved pose: base=%.0f shoulder=%.0f elbow=%.0f gripper=%.0f\n", saved[0], saved[1], saved[2], saved[3]);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(PIN_J1_SW, INPUT);
  pinMode(PIN_J2_SW, INPUT);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  prefs.begin("arm", false);
  hadSaved = prefs.isKey(KEYS[0]);
  for (int i = 0; i < 4; i++) {
    saved[i] = constrain(prefs.getFloat(KEYS[i], START_ANGLE), (float)ANGLE_MIN, (float)ANGLE_MAX);
    cmd[i] = saved[i];
    tgt[i] = START_ANGLE;
  }
  if (hadSaved) {
    Serial.printf("last saved pose: base=%.0f shoulder=%.0f elbow=%.0f gripper=%.0f\n", cmd[0], cmd[1], cmd[2], cmd[3]);
  } else {
    Serial.println("no saved pose yet - first boot will jump to 90, set the joints near 90 by hand");
  }

  Serial.println("Calibrating joystick centers - do not touch the sticks...");
  centerJ1x = calibrateCenter(PIN_J1_X);
  centerJ1y = calibrateCenter(PIN_J1_Y);
  centerJ2x = calibrateCenter(PIN_J2_X);
  centerJ2y = calibrateCenter(PIN_J2_Y);
  Serial.printf("Centers: J1x=%d J1y=%d J2x=%d J2y=%d\n", centerJ1x, centerJ1y, centerJ2x, centerJ2y);

  for (int i = 0; i < 4; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(SERVO_PINS[i], PULSE_MIN_US, PULSE_MAX_US);
    servos[i].write((int)round(cmd[i]));
    delay(200);
  }

  Serial.printf("moving slowly to %d deg at %.0f deg/s - sticks are ignored until then\n", START_ANGLE, STARTUP_DEG_PER_SEC);
  lastUpdateMs = millis();
  lastMoveMs = lastUpdateMs;
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

  if (!starting && !frozen) {
    tgt[0] += applyDeadzone(j1x, centerJ1x) * DEG_PER_SEC_AT_FULL_DEFLECTION * dtSec;
    tgt[1] += applyDeadzone(j1y, centerJ1y) * DEG_PER_SEC_AT_FULL_DEFLECTION * dtSec;
    tgt[2] += applyDeadzone(j2x, centerJ2x) * DEG_PER_SEC_AT_FULL_DEFLECTION * dtSec;
    tgt[3] = map(j2y, 0, 4095, ANGLE_MIN, ANGLE_MAX);
  }

  float speed = starting ? STARTUP_DEG_PER_SEC : MAX_DEG_PER_SEC;
  float maxStep = speed * dtSec;
  bool allThere = true;
  for (int i = 0; i < 4; i++) {
    tgt[i] = constrain(tgt[i], (float)ANGLE_MIN, (float)ANGLE_MAX);
    float d = tgt[i] - cmd[i];
    if (fabs(d) > 0.5) {
      allThere = false;
      lastMoveMs = now;
    }
    cmd[i] += constrain(d, -maxStep, maxStep);
    servos[i].write((int)round(cmd[i]));
  }

  if (starting && allThere) {
    starting = false;
    Serial.println("at start pose - joystick control ON");
  }

  saveIfSettled(now);

  if (now - lastPrintMs > 200) {
    lastPrintMs = now;
    Serial.printf("J1 x=%d y=%d sw=%d | J2 x=%d y=%d sw=%d | base=%.1f shoulder=%.1f elbow=%.1f gripper=%.1f%s%s\n",
                  j1x, j1y, j1sw, j2x, j2y, j2sw, cmd[0], cmd[1], cmd[2], cmd[3],
                  starting ? " [STARTING]" : "", frozen ? " [FROZEN]" : "");
    for (int i = 0; i < 3; i++) warnIfAtLimit(i);
  }
}
