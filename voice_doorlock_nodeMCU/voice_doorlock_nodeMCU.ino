// ESP8266 Doorlock - Minimal (AP 모드 + TCP 제어 + 릴레이 + SEQ 상태머신)
// 파이썬에서 "AUTHOK" 또는 "SEQ on1 gap on2" 한 줄만 보냄.

#include <ESP8266WiFi.h>

// ── 네트워크 ──
#define USE_AP_MODE 1              // 1: SoftAP, 0: STA(공유기 접속)
#if USE_AP_MODE
const char* ap_ssid = "DoorlockAP";
const char* ap_pass = "12345678";  // 8자 이상
#else
const char* sta_ssid = "";
const char* sta_pass = "";
#endif
WiFiServer server(7777);

// ── 릴레이 핀/논리 설정 ──
#define RELAY_PIN D1               // GPIO5 → 릴레이 IN
#define RELAY_ACTIVE_LOW 1         // 대부분 릴레이 모듈은 'LOW일 때 ON'

// ── 상태 미러용 LED (D6, Active-High) ──
#define MIRROR_LED 1
#define MIRROR_LED_PIN D6
#define MIRROR_LED_ACTIVE_HIGH 1   // HIGH=켜짐

// ── OPEN(단일 펄스) 상태 ──
bool relayOn = false;
bool pulseInProgress = false;
unsigned long pulseEndAt = 0;

// ── SEQ 상태머신 ──
bool seqActive = false;
uint8_t seqPhase = 0;              // 0=idle, 1=on1, 2=gap(off), 3=on2
unsigned long seqPhaseEndAt = 0;
int seqOn1 = 0, seqGap = 0, seqOn2 = 0;
const int SEQ_MIN_MS = 10;
const int SEQ_MAX_MS = 20000;

// ── 유틸 ──
void logln(const String& s) { Serial.println(s); }

void setMirrorLed(bool on) {
#if MIRROR_LED
  pinMode(MIRROR_LED_PIN, OUTPUT);
#if MIRROR_LED_ACTIVE_HIGH
  digitalWrite(MIRROR_LED_PIN, on ? HIGH : LOW);
#else
  digitalWrite(MIRROR_LED_PIN, on ? LOW : HIGH);
#endif
#endif
}

void setRelay(bool on) {
  relayOn = on;
#if RELAY_ACTIVE_LOW
  digitalWrite(RELAY_PIN, on ? LOW : HIGH);
#else
  digitalWrite(RELAY_PIN, on ? HIGH : LOW);
#endif
  setMirrorLed(on);
}

int relayStateForReport() { return relayOn ? 1 : 0; }

String readLineFromClient(WiFiClient& c) {
  static String buf;
  while (c.available()) {
    char ch = c.read();
    if (ch == '\r') continue;
    if (ch == '\n') { String line = buf; buf = ""; return line; }
    buf += ch;
  }
  return String();
}

String readLineFromSerial() {
  static String sbuf;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') { String line = sbuf; sbuf = ""; return line; }
    sbuf += ch;
  }
  return String();
}

void reply(WiFiClient* pc, const String& s) {
  if (pc && pc->connected()) pc->println(s);
  logln(s);
}

// ── SEQ 취소 ──
void cancelSeqIfRunning() {
  if (seqActive) {
    seqActive = false;
    seqPhase = 0;
    setRelay(false);
    logln("SEQ CANCELED");
  }
}

// ── SEQ 시작 ──
void startSeq(int on1, int gap, int on2, WiFiClient* pc) {
  on1 = constrain(on1, SEQ_MIN_MS, SEQ_MAX_MS);
  gap = constrain(gap, SEQ_MIN_MS, SEQ_MAX_MS);
  on2 = constrain(on2, SEQ_MIN_MS, SEQ_MAX_MS);

  // 다른 동작 취소
  pulseInProgress = false;
  cancelSeqIfRunning();

  seqOn1 = on1; seqGap = gap; seqOn2 = on2;
  seqActive = true; seqPhase = 1;
  setRelay(true);
  seqPhaseEndAt = millis() + (unsigned long)seqOn1;

  reply(pc, "OK SEQ " + String(seqOn1) + " " + String(seqGap) + " " + String(seqOn2));
  reply(pc, "STATUS " + String(relayStateForReport()));
}

// ── 명령 처리 ──
void processCommand(const String& raw, WiFiClient* pc) {
  String line = raw; line.trim(); line.toUpperCase();

  if (line == "PING") {
    reply(pc, "PONG");
  }
  else if (line.startsWith("OPEN")) {  // OPEN [ms]
    cancelSeqIfRunning();
    int ms = 500;
    int sp = line.indexOf(' ');
    if (sp > 0) { int v = line.substring(sp + 1).toInt(); if (v > 0) ms = v; }
    const int MIN_PULSE_MS = 50, MAX_PULSE_MS = 10000;
    ms = constrain(ms, MIN_PULSE_MS, MAX_PULSE_MS);

    setRelay(true);
    pulseInProgress = true;
    pulseEndAt = millis() + (unsigned long)ms;

    reply(pc, "OK OPEN " + String(ms));
    reply(pc, "STATUS " + String(relayStateForReport()));
  }
  else if (line == "CLOSE") {
    cancelSeqIfRunning();
    setRelay(false);
    pulseInProgress = false;
    reply(pc, "OK CLOSE");
    reply(pc, "STATUS " + String(relayStateForReport()));
  }
  else if (line == "STATUS") {
    reply(pc, "STATUS " + String(relayStateForReport()));
  }
  else if (line.startsWith("SEQ")) {   // SEQ on1 gap on2
    int on1 = 1000, gap = 5000, on2 = 1000;   // 기본값
    int p1 = line.indexOf(' ');
    if (p1 > 0) {
      String rest = line.substring(p1 + 1);
      int p2 = rest.indexOf(' ');
      int p3 = (p2 > 0) ? rest.indexOf(' ', p2 + 1) : -1;
      if (p2 > 0 && p3 > 0) {
        on1 = rest.substring(0, p2).toInt();
        gap = rest.substring(p2 + 1, p3).toInt();
        on2 = rest.substring(p3 + 1).toInt();
      }
    }
    startSeq(on1, gap, on2, pc);
  }
  else if (line == "AUTHOK") {        // 편의: 1초 ON → 5초 OFF → 1초 ON
    startSeq(1000, 5000, 1000, pc);
  }
  else {
    reply(pc, "ERR UNKNOWN: " + line);
  }
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  setRelay(false);
#if MIRROR_LED
  pinMode(MIRROR_LED_PIN, OUTPUT);
  setMirrorLed(false);
#endif

  Serial.begin(115200);
  delay(200);

#if USE_AP_MODE
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ap_ssid, ap_pass);
  IPAddress ip = WiFi.softAPIP();
  logln("READY AP " + ip.toString());   // 보통 192.168.4.1
#else
  WiFi.mode(WIFI_STA);
  WiFi.begin(sta_ssid, sta_pass);
  Serial.print("WIFI CONNECTING");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println();
  logln("READY STA " + WiFi.localIP().toString());
#endif

  server.begin();
  logln("SERVER 7777 STARTED");
  logln(RELAY_ACTIVE_LOW ? "RELAY LOGIC: ACTIVE_LOW" : "RELAY LOGIC: ACTIVE_HIGH");
}

void loop() {
  unsigned long now = millis();

  // ── OPEN(단일 펄스) 종료 ──
  if (pulseInProgress && (long)(now - pulseEndAt) >= 0) {
    setRelay(false);
    pulseInProgress = false;
    logln("PULSE DONE");
    logln("STATUS " + String(relayStateForReport()));
  }

  // ── SEQ 상태머신 ──
  if (seqActive && (long)(now - seqPhaseEndAt) >= 0) {
    if (seqPhase == 1) {                // on1 끝 → gap
      setRelay(false);
      seqPhase = 2;
      seqPhaseEndAt = now + (unsigned long)seqGap;
      logln("SEQ GAP");
      logln("STATUS " + String(relayStateForReport()));
    } else if (seqPhase == 2) {         // gap 끝 → on2
      setRelay(true);
      seqPhase = 3;
      seqPhaseEndAt = now + (unsigned long)seqOn2;
      logln("SEQ ON2");
      logln("STATUS " + String(relayStateForReport()));
    } else if (seqPhase == 3) {         // on2 끝 → 종료
      setRelay(false);
      seqActive = false;
      seqPhase = 0;
      logln("SEQ DONE");
      logln("STATUS " + String(relayStateForReport()));
    }
  }

  // ── TCP 클라이언트 처리 ──
  static WiFiClient client;
  if (!client || !client.connected()) {
    client = server.available();
    if (client) logln("CLIENT CONNECTED");
  } else {
    String cmd = readLineFromClient(client);
    if (cmd.length() > 0) processCommand(cmd, &client);
    if (!client.connected()) {
      client.stop();
      logln("CLIENT DISCONNECT");
    }
  }

  // (옵션) 시리얼로도 테스트 가능
  String scmd = readLineFromSerial();
  if (scmd.length() > 0) processCommand(scmd, nullptr);

  yield(); // WiFi 유지
}
