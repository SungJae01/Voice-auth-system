// WiFi를 사용하여 노트북에서 실행한 Python 코드와 통신
// 인증 과정이 모두 완료되면 7초 동안 도어락 개방 후 자동으로 닫힘

#include <ESP8266WiFi.h>

// ── 네트워크: 옵션 A (파이썬은 TCP로 명령 / USB 시리얼은 모니터 전용) ──
#define USE_AP_MODE 0              // 1: SoftAP, 0: STA(공유기 접속)
#if USE_AP_MODE
const char* ap_ssid = "DoorlockAP";
const char* ap_pass = "12345678";  // 8자 이상
#else
const char* sta_ssid = "WIFI_SSID";
const char* sta_pass = "PASSWORD";
#endif
WiFiServer server(7777);

// ── 릴레이 핀/논리 설정 ──
#define RELAY_PIN D1               // GPIO5 (권장) D1 핀으로 릴레이를 제어하여 도어락을 열고 닫음
#define RELAY_ACTIVE_LOW 1         // 대부분 릴레이 모듈은 'LOW일 때 ON' 
#define MIRROR_LED 1               // 보드 LED(D4, active-low)로 상태 미러 (디버깅용)

// ── 펄스 타이머 상태 ──
bool relayOn = false;
bool pulseInProgress = false;
unsigned long pulseEndAt = 0;

// ── 유틸 ──
void logln(const String& s) { Serial.println(s); }

void setRelay(bool on) {
  relayOn = on;
#if RELAY_ACTIVE_LOW
  digitalWrite(RELAY_PIN, on ? LOW : HIGH);
#else
  digitalWrite(RELAY_PIN, on ? HIGH : LOW);
#endif
#if MIRROR_LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, on ? LOW : HIGH); // 보드 LED는 active-low
#endif
}

int relayStateForReport() {
  // 논리에 상관없이 "현재 에너지 인가 여부"를 1/0으로 리턴
  return relayOn ? 1 : 0;
}

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

void replyBoth(WiFiClient* pc, const String& s) {
  if (pc && pc->connected()) pc->println(s);
  logln(s); // 항상 시리얼 모니터에도 출력
}

void processCommand(const String& raw, WiFiClient* pc) {
  String line = raw; line.trim(); line.toUpperCase();

  if (line == "PING") {
    replyBoth(pc, "PONG");
  }
  else if (line.startsWith("OPEN")) {
    // 형식: OPEN 700  (700ms 펄스)
    int ms = 500;
    int sp = line.indexOf(' ');
    if (sp > 0) {
      int v = line.substring(sp + 1).toInt();
      if (v > 0) ms = v;
    }
    const int MIN_PULSE_MS = 50;     // 너무 짧은 펄스 보호
    const int MAX_PULSE_MS = 10000;   // 과도 펄스 보호
    ms = constrain(ms, MIN_PULSE_MS, MAX_PULSE_MS);

    setRelay(true);
    pulseInProgress = true;
    pulseEndAt = millis() + (unsigned long)ms;

    replyBoth(pc, "OK OPEN " + String(ms));
    replyBoth(pc, "STATUS " + String(relayStateForReport())); // 즉시 상태도 한번 찍기
  }
  else if (line == "CLOSE") {
    setRelay(false);
    pulseInProgress = false;
    replyBoth(pc, "OK CLOSE");
    replyBoth(pc, "STATUS " + String(relayStateForReport()));
  }
  else if (line == "STATUS") {
    replyBoth(pc, "STATUS " + String(relayStateForReport()));
  }
  else {
    replyBoth(pc, "ERR UNKNOWN: " + line);
  }
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  setRelay(false);                  // 초기 OFF
  Serial.begin(115200);
  delay(200);

#if USE_AP_MODE
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ap_ssid, ap_pass);
  IPAddress ip = WiFi.softAPIP();
  logln("READY AP " + ip.toString());          // 예: 192.168.4.1
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
  // ── 1) 펄스 종료 처리 (논블로킹) ──
  if (pulseInProgress && (long)(millis() - pulseEndAt) >= 0) {
    setRelay(false);
    pulseInProgress = false;
    logln("PULSE DONE");
    logln("STATUS " + String(relayStateForReport()));
  }

  // ── 2) TCP 클라이언트 처리 ──
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

  // ── 3) (선택) 시리얼 모니터에서 직접 명령 입력해도 동일 동작 ──
  String scmd = readLineFromSerial();
  if (scmd.length() > 0) {
    // 시리얼로 들어온 명령은 네트워크 응답 없이 로그만(=시리얼 모니터) 출력
    processCommand(scmd, nullptr);
  }

  yield(); // 와이파이 유지
}
