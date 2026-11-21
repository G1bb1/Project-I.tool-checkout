/*
  toolcheck_watch_v2: CHECKOUT and SIGN-OFF from the watch, board status back
  Board: Waveshare ESP32-S3-Touch-AMOLED-2.06 (Arduino core for ESP32)
  Library: "GFX Library for Arduino" (moononournation), same as v1
  Board settings: ESP32S3 Dev Module, USB CDC On Boot enabled,
                  Flash Size 32MB, PSRAM OPI PSRAM
  PC side: python watch_main.py serve        (or serve --replay data/<session>)

  Changes from v1 (toolcheck_watch.ino)
    * two commands instead of one "start":
        checkout -> PC takes the checkout photo (the Method A baseline), the model
                    checks every slot holds its own tool. CHECKOUT OK or NOT CHECKED OUT.
        signoff  -> PC compares with the checkout photo (hybrid: Method A + model).
                    KIT CONFIRMED / MISSING + tools / RETAKE + reason / NO CHECKOUT.
    * the reply carries the board status: present/total and the tools that are not right.
    * the BOOT button does the next step of the job: checkout when no job is open,
      sign-off when one is open. Hold it 2 s to force a new checkout.
    * at start-up the watch asks the PC whether a job is open ("status").
    * replies up to 1023 bytes.
  Unchanged: 10 s with no valid answer = NOT CONFIRMED (REQ-TR-04); a reply for an
  older check, a late reply, or an unknown status never shows as a pass.

  Serial Monitor (115200 baud):
    c        checkout           s       sign-off          q   ask the PC: job open?
    a60      60 sign-offs back to back, one every 12 s  (server OFF = the 60/60 timeout run)
    j10      10 jobs: checkout, then sign-off (again on RETAKE, up to 3 times)  (replay run)
    x        stop auto mode     r       print the counts

  Every finished exchange prints one evidence line:
    LOG,<id>,<op>,<start_ms>,<outcome>,<latency_ms>,<detail>
  outcome: checkout_ok | checkout_refused | checkout_timeout | confirmed | missing |
           retake | no_checkout | not_confirmed_timeout | not_confirmed_bad_reply |
           status_job_open | status_idle
  latency_ms = tap -> PC/processor -> answer on the watch (end to end).
  Save the Serial Monitor output as a .txt, then: python watch_main.py report ... --serial <file>

  Display note: the CO5300 only draws areas that start on an even pixel with even
  width/height, so positions are even and text sizes are 2 and 4 only.
  Titles are at most 15 characters (15 x 24 px + margin fits the 410 px width).
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Arduino_GFX_Library.h>

// ------------------------------- EDIT THESE -------------------------------
const char* WIFI_SSID = "Helix_84";
const char* WIFI_PASS = "C9WMA9EMPYTYJ";
IPAddress   PC_IP(10, 0, 0, 225);      // the PC (or Pi) running: python watch_main.py serve
const uint16_t PC_PORT    = 4211;        // settings.json -> watch_server.listen_port
const uint16_t WATCH_PORT = 4210;        // the watch listens here for answers
const uint32_t TIMEOUT_MS = 10000;       // REQ-TR-04: 10 s
const uint32_t AUTO_GAP_MS = 12000;      // a60: time between sign-offs (> timeout)
const uint32_t JOB_GAP_MS  = 3000;       // j:   time between steps of a job
const int START_BUTTON_PIN = 0;          // BOOT button, active LOW; -1 = no button
const uint32_t LONG_PRESS_MS = 2000;     // hold = force a new checkout
// ---------------------------------------------------------------------------

// ---------- Display pins (Waveshare wiki) ----------
#define LCD_SDIO0  4
#define LCD_SDIO1  5
#define LCD_SDIO2  6
#define LCD_SDIO3  7
#define LCD_SCLK   11
#define LCD_CS     12
#define LCD_RESET  8
#define LCD_WIDTH  410
#define LCD_HEIGHT 502

// ---------- Colors (RGB565) ----------
#define C_BLACK  0x0000
#define C_WHITE  0xFFFF
#define C_RED    0xF800
#define C_GREEN  0x07E0
#define C_AMBER  0xFD20
#define C_GREY   0x7BEF

Arduino_DataBus *bus = new Arduino_ESP32QSPI(
    LCD_CS, LCD_SCLK, LCD_SDIO0, LCD_SDIO1, LCD_SDIO2, LCD_SDIO3);
Arduino_CO5300 *gfx = new Arduino_CO5300(
    bus, LCD_RESET, 0 /* rotation */, LCD_WIDTH, LCD_HEIGHT,
    22, 0, 0, 0 /* column/row offsets */);

const int MARGIN   = 24;   // keeps text away from the rounded corners (even)
const int TITLE_Y  = 120;  // title line (even)
const int DETAIL_Y = 200;  // first detail line (even)
const int LINE_H   = 24;   // one detail line at text size 2 (even)

enum State { IDLE, WAITING };
enum Op { OP_CHECKOUT, OP_SIGNOFF, OP_STATUS };
const char* OP_NAME[] = {"checkout", "signoff", "status"};

WiFiUDP udp;
State state = IDLE;
Op pendingOp = OP_SIGNOFF;
uint32_t checkId = 0;
uint32_t startMs = 0;
bool jobOpen = false;          // what the watch believes; the PC's answer always wins

// auto modes
int autoSignoffs = 0;          // a60
int autoJobs = 0;              // j10
int jobRetries = 0;
Op nextJobOp = OP_CHECKOUT;
uint32_t nextAutoMs = 0;

uint32_t nConfirmed = 0, nMissing = 0, nNotConfirmed = 0, nStale = 0;
uint32_t nCheckoutOk = 0, nCheckoutRefused = 0, nCheckoutTimeout = 0, nRetake = 0, nNoCheckout = 0;

uint32_t buttonDownMs = 0;
bool buttonWasDown = false, longFired = false;
uint32_t lastWifiTryMs = 0;
String lastOutcome = "";

// ---------------------------------------------------------------------------
// Screen: title large (size 4), detail small (size 2), detail lines split on ';'.
void showStatus(const char* title, const String& detail, uint16_t titleColor = C_WHITE,
                uint16_t bg = C_BLACK, uint16_t detailColor = C_GREY) {
  gfx->fillScreen(bg);
  gfx->setTextWrap(false);
  gfx->setTextSize(4);
  gfx->setTextColor(titleColor);
  gfx->setCursor(MARGIN, TITLE_Y);
  gfx->print(title);

  gfx->setTextSize(2);
  gfx->setTextColor(detailColor);
  int y = DETAIL_Y;
  int from = 0;
  while (from <= (int)detail.length() && y < LCD_HEIGHT - 40) {
    int semi = detail.indexOf(';', from);
    String line = (semi < 0) ? detail.substring(from) : detail.substring(from, semi);
    line.trim();
    if (line.length() > 29) line = line.substring(0, 28) + "~";   // fit the width
    if (line.length() > 0) {
      gfx->setCursor(MARGIN, y);
      gfx->print(line);
      y += LINE_H;
    }
    if (semi < 0) break;
    from = semi + 1;
  }
  Serial.printf("[SCREEN] %s | %s\n", title, detail.c_str());
}

// The fail screen is white so it cannot be mistaken for a result screen.
void showNotConfirmed(const char* title, const String& detail) {
  showStatus(title, detail, C_BLACK, C_WHITE, C_BLACK);
}

void logLine(uint32_t id, Op op, uint32_t start, const char* outcome, long latency, const String& detail) {
  String d = detail;
  d.replace("\n", " ");
  Serial.printf("LOG,%lu,%s,%lu,%s,%ld,%s\n", (unsigned long)id, OP_NAME[op], (unsigned long)start,
                outcome, latency, d.c_str());
  lastOutcome = outcome;
}

// --- tiny JSON helpers (the PC sends one flat object; tool names have no quotes) ---
long jsonInt(const String& s, const char* key) {
  String k = String("\"") + key + "\":";
  int i = s.indexOf(k);
  if (i < 0) return -1;
  i += k.length();
  while (i < (int)s.length() && s[i] == ' ') i++;
  if (i >= (int)s.length() || !(isDigit(s[i]) || s[i] == '-')) return -1;
  return s.substring(i).toInt();
}

String jsonStr(const String& s, const char* key) {
  String k = String("\"") + key + "\":";
  int i = s.indexOf(k);
  if (i < 0) return "";
  i = s.indexOf('"', i + k.length());
  if (i < 0) return "";
  int j = s.indexOf('"', i + 1);
  if (j < 0) return "";
  return s.substring(i + 1, j);
}

String jsonList(const String& s, const char* key) {   // ["a","b"] -> a; b
  String k = String("\"") + key + "\":";
  int i = s.indexOf(k);
  if (i < 0) return "";
  int a = s.indexOf('[', i), b = s.indexOf(']', a);
  if (a < 0 || b < 0) return "";
  String inner = s.substring(a + 1, b);
  inner.replace("\", \"", "; ");
  inner.replace("\",\"", "; ");
  inner.replace("\"", "");
  return inner;
}

// ---------------------------------------------------------------------------
void sendCmd(Op op) {
  if (state == WAITING) {
    Serial.println("A check is already running, ignored.");
    return;
  }
  checkId++;
  startMs = millis();
  pendingOp = op;
  state = WAITING;
  if (op == OP_CHECKOUT) showStatus("CHECKING OUT", String("check ") + checkId);
  else if (op == OP_SIGNOFF) showStatus("CHECKING...", String("check ") + checkId);
  // Sent even if Wi-Fi is down: the check must then time out, never pass.
  if (WiFi.status() == WL_CONNECTED) {
    char msg[64];
    snprintf(msg, sizeof(msg), "{\"cmd\":\"%s\",\"id\":%lu}", OP_NAME[op], (unsigned long)checkId);
    udp.beginPacket(PC_IP, PC_PORT);
    udp.write((const uint8_t*)msg, strlen(msg));
    udp.endPacket();
  }
}

void handlePacket() {
  int len = udp.parsePacket();
  if (len <= 0) return;
  static char buf[1024];
  int n = udp.read(buf, sizeof(buf) - 1);
  if (n <= 0) return;
  buf[n] = '\0';
  String s(buf);

  long id = jsonInt(s, "id");
  String op = jsonStr(s, "op");
  if (state != WAITING || id != (long)checkId || op != OP_NAME[pendingOp]) {
    nStale++;
    Serial.printf("Ignored late/stale answer for id %ld (current %lu, %s)\n",
                  id, (unsigned long)checkId, state == WAITING ? "waiting" : "idle");
    return;
  }
  long latency = (long)(millis() - startMs);
  String status = jsonStr(s, "status");
  String tools = jsonList(s, "tools");
  String reason = jsonStr(s, "reason");
  long present = jsonInt(s, "present"), total = jsonInt(s, "total"), job = jsonInt(s, "job");
  String count = (present >= 0 && total >= 0) ? String(present) + "/" + String(total) + " tools in place" : "";
  state = IDLE;

  if (pendingOp == OP_CHECKOUT) {
    if (status == "ready") {
      nCheckoutOk++;
      jobOpen = true;
      String absent = jsonList(s, "absent");
      showStatus("CHECKOUT OK", String("job ") + job + ";" + count +
                 (absent.length() ? ";not on board: " + absent : ""), C_GREEN);
      logLine(checkId, pendingOp, startMs, "checkout_ok", latency, count);
    } else if (status == "refused") {
      nCheckoutRefused++;
      jobOpen = false;
      showStatus("NOT CHECKED OUT", String("fix the board first;") + tools, C_AMBER, C_BLACK, C_WHITE);
      logLine(checkId, pendingOp, startMs, "checkout_refused", latency, tools);
    } else {
      nNotConfirmed++;
      showNotConfirmed("NOT CHECKED OUT", String("bad reply;") + status + " " + reason);
      logLine(checkId, pendingOp, startMs, "not_confirmed_bad_reply", latency, status + " " + reason);
    }
    return;
  }

  if (pendingOp == OP_STATUS) {
    if (status == "job_open" || status == "idle") {
      jobOpen = (status == "job_open");
      showStatus("READY", jobOpen ? String("job ") + job + " open;press = sign off"
                                  : String("no job open;press = check out"));
      logLine(checkId, pendingOp, startMs, jobOpen ? "status_job_open" : "status_idle", latency, "");
    } else {
      showStatus("READY", "PC answer unclear;press = check out");
      logLine(checkId, pendingOp, startMs, "not_confirmed_bad_reply", latency, status);
    }
    return;
  }

  // sign-off: only "complete" is a pass
  if (status == "complete") {
    nConfirmed++;
    jobOpen = false;
    showStatus("KIT CONFIRMED", count + ";" + String(latency) + " ms", C_GREEN);
    logLine(checkId, pendingOp, startMs, "confirmed", latency, count);
  } else if (status == "missing") {
    nMissing++;
    showStatus("MISSING", tools + ";" + count, C_RED, C_BLACK, C_WHITE);
    logLine(checkId, pendingOp, startMs, "missing", latency, tools);
  } else if (status == "retake") {
    nRetake++;
    // also show what the check already sees, so a missing tool is never hidden behind a retake
    showStatus("RETAKE", reason + ";press again" + (tools.length() ? String(";sees: ") + tools : String("")),
               C_AMBER, C_BLACK, C_WHITE);
    logLine(checkId, pendingOp, startMs, "retake", latency, reason);
  } else if (status == "no_checkout") {
    nNoCheckout++;
    jobOpen = false;
    showNotConfirmed("NO CHECKOUT", "no open job on the PC;check out first");
    logLine(checkId, pendingOp, startMs, "no_checkout", latency, "");
  } else {
    nNotConfirmed++;
    showNotConfirmed("NOT CONFIRMED", String("bad reply;") + status + " " + reason);
    logLine(checkId, pendingOp, startMs, "not_confirmed_bad_reply", latency, status + " " + reason);
  }
}

void checkTimeout() {
  if (state == WAITING && millis() - startMs >= TIMEOUT_MS) {
    state = IDLE;
    long lat = (long)(millis() - startMs);
    if (pendingOp == OP_CHECKOUT) {
      nCheckoutTimeout++;
      showNotConfirmed("NOT CHECKED OUT", "no answer in 10 s");
      logLine(checkId, pendingOp, startMs, "checkout_timeout", lat, "");
    } else if (pendingOp == OP_SIGNOFF) {
      nNotConfirmed++;
      showNotConfirmed("NOT CONFIRMED", "no result in 10 s;count the kit by hand");
      logLine(checkId, pendingOp, startMs, "not_confirmed_timeout", lat, "");
    } else {
      showStatus("READY", "PC not answering;press = check out");
      logLine(checkId, pendingOp, startMs, "not_confirmed_timeout", lat, "");
    }
  }
}

void printCounts() {
  Serial.printf("COUNTS,checks=%lu,checkout_ok=%lu,checkout_refused=%lu,checkout_timeout=%lu,"
                "confirmed=%lu,missing=%lu,retake=%lu,no_checkout=%lu,not_confirmed=%lu,stale_ignored=%lu\n",
                (unsigned long)checkId, (unsigned long)nCheckoutOk, (unsigned long)nCheckoutRefused,
                (unsigned long)nCheckoutTimeout, (unsigned long)nConfirmed, (unsigned long)nMissing,
                (unsigned long)nRetake, (unsigned long)nNoCheckout, (unsigned long)nNotConfirmed,
                (unsigned long)nStale);
}

void handleSerial() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd == "c") sendCmd(OP_CHECKOUT);
  else if (cmd == "s") sendCmd(OP_SIGNOFF);
  else if (cmd == "q") sendCmd(OP_STATUS);
  else if (cmd.startsWith("a")) {
    autoSignoffs = cmd.substring(1).toInt();
    autoJobs = 0;
    nextAutoMs = millis();
    Serial.printf("Auto: %d sign-offs, one every %lu ms\n", autoSignoffs, (unsigned long)AUTO_GAP_MS);
  } else if (cmd.startsWith("j")) {
    autoJobs = cmd.substring(1).toInt();
    autoSignoffs = 0;
    nextJobOp = OP_CHECKOUT;
    jobRetries = 0;
    nextAutoMs = millis();
    Serial.printf("Auto: %d jobs (checkout, then sign-off)\n", autoJobs);
  } else if (cmd == "x") {
    autoSignoffs = 0;
    autoJobs = 0;
    Serial.println("Auto mode stopped.");
  } else if (cmd == "r") printCounts();
}

void handleAuto() {
  if (state != IDLE || (long)(millis() - nextAutoMs) < 0) return;
  if (autoSignoffs > 0) {
    autoSignoffs--;
    nextAutoMs = millis() + AUTO_GAP_MS;
    sendCmd(OP_SIGNOFF);
    if (autoSignoffs == 0) Serial.println("Auto: last sign-off started.");
    return;
  }
  if (autoJobs > 0) {
    // decide the next step from the last outcome
    if (nextJobOp == OP_SIGNOFF && lastOutcome == "retake" && jobRetries < 3) {
      jobRetries++;
      nextAutoMs = millis() + JOB_GAP_MS;
      sendCmd(OP_SIGNOFF);
      return;
    }
    if (nextJobOp == OP_CHECKOUT) {
      jobRetries = 0;
      nextJobOp = OP_SIGNOFF;
      nextAutoMs = millis() + JOB_GAP_MS;
      sendCmd(OP_CHECKOUT);
      return;
    }
    // first sign-off of this job, or the job just finished
    if (lastOutcome == "checkout_ok" || lastOutcome == "checkout_refused" || lastOutcome == "checkout_timeout") {
      nextAutoMs = millis() + JOB_GAP_MS;
      sendCmd(OP_SIGNOFF);
      return;
    }
    autoJobs--;
    nextJobOp = OP_CHECKOUT;
    nextAutoMs = millis() + JOB_GAP_MS;
    if (autoJobs == 0) Serial.println("Auto: all jobs done.");
  }
}

// Short press: the next step of the job. Hold 2 s: new checkout.
void handleButton() {
  if (START_BUTTON_PIN < 0) return;
  bool down = digitalRead(START_BUTTON_PIN) == LOW;
  uint32_t now = millis();
  if (down && !buttonWasDown) {
    buttonDownMs = now;
    longFired = false;
  }
  if (down && !longFired && now - buttonDownMs >= LONG_PRESS_MS) {
    longFired = true;
    sendCmd(OP_CHECKOUT);
  }
  if (!down && buttonWasDown && !longFired && now - buttonDownMs > 30) {
    sendCmd(jobOpen ? OP_SIGNOFF : OP_CHECKOUT);
  }
  buttonWasDown = down;
}

void keepWifi() {
  if (WiFi.status() != WL_CONNECTED && millis() - lastWifiTryMs > 5000) {
    lastWifiTryMs = millis();
    WiFi.reconnect();   // non-blocking; checks still time out while it is down
  }
}

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(1500);
  if (START_BUTTON_PIN >= 0) pinMode(START_BUTTON_PIN, INPUT_PULLUP);

  if (!gfx->begin()) Serial.println("Display init failed!");
  gfx->setBrightness(200);     // 0-255

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  showStatus("CONNECTING", WIFI_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) delay(200);
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("Wi-Fi OK, watch IP %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("Wi-Fi not connected yet (checks will time out until it is).");
  }
  udp.begin(WATCH_PORT);
  showStatus("READY", "press = check out;c / s / q in Serial");
  Serial.println("LOG,id,op,start_ms,outcome,latency_ms,detail");
  sendCmd(OP_STATUS);          // is a job already open on the PC?
}

void loop() {
  handleSerial();
  handleButton();
  handleAuto();
  handlePacket();
  checkTimeout();
  keepWifi();
  delay(5);
}
