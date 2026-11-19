/*
 * Wi-Fi Scan & Connect
 * Board: Waveshare ESP32-S3-Touch-AMOLED-2.06
 *
 * Scans for Wi-Fi networks in the background and lists the strongest ones.
 * Tap a network to connect to it with the stored password.
 *   Success: black screen, white inverted triangle builds from the bottom up,
 *            then cycles between white and dark grey.
 *   Failure: white screen, black triangle pointing up builds from the bottom up.
 * Tap anywhere to go back to the list.
 *
 * IMPORTANT (CO5300 display): every drawing area must start on an even pixel
 * and be an even number of pixels wide/tall, otherwise the chip ignores it.
 * All drawing goes through rect2() or even-sized text at even positions.
 *
 * Library:        GFX Library for Arduino (by Moon On Our Nation)
 * Board settings: ESP32S3 Dev Module, USB CDC On Boot: Enabled,
 *                 Flash Size: 32MB, PSRAM: OPI PSRAM
 */

#include <WiFi.h>
#include <Wire.h>
#include <Arduino_GFX_Library.h>
#include <algorithm>

// =====================================================================
//  Hardware pins (from the Waveshare docs)
// =====================================================================
namespace Pin {
  // Display (QSPI, CO5300)
  constexpr int LCD_SDIO0 = 4;
  constexpr int LCD_SDIO1 = 5;
  constexpr int LCD_SDIO2 = 6;
  constexpr int LCD_SDIO3 = 7;
  constexpr int LCD_SCLK  = 11;
  constexpr int LCD_CS    = 12;
  constexpr int LCD_RESET = 8;

  // Touch (shared I2C bus, FT3168)
  constexpr int TP_SDA = 15;
  constexpr int TP_SCL = 14;
  constexpr int TP_RST = 9;
}

// =====================================================================
//  Settings
// =====================================================================
namespace Cfg {
  // Password used for any secured network you tap (open networks need none).
  // Don't push this file to a public repo with the password in it.
  constexpr const char *WIFI_PASSWORD = "nttg0065";

  constexpr int      SCREEN_W   = 410;
  constexpr int      SCREEN_H   = 502;
  constexpr int      COL_OFFSET = 22;     // shift if the image looks cut off
  constexpr uint8_t  BRIGHTNESS = 200;    // 0-255

  constexpr uint8_t  TOUCH_ADDR = 0x38;   // FT3168 I2C address

  constexpr unsigned long SCAN_INTERVAL_MS   = 10000; // pause between scans (calmer list)
  constexpr unsigned long CONNECT_TIMEOUT_MS = 15000; // give up connecting after this
  constexpr unsigned long LOOP_DELAY_MS      = 20;    // ~50 touch checks per second
  constexpr unsigned long ROW_FLASH_MS       = 150;   // how long a tapped row highlights
  constexpr int           MAX_RESULTS        = 40;    // scan results kept for sorting
}

// =====================================================================
//  Screen layout (keep every number even - see note at the top)
// =====================================================================
namespace Layout {
  constexpr int MARGIN   = 24;    // keeps content away from the rounded corners
  constexpr int TITLE_Y  = 36;
  constexpr int STATUS_Y = 80;
  constexpr int HINT_Y   = 102;

  constexpr int LIST_Y   = 130;   // top of the first row
  constexpr int ROW_H    = 34;    // tall rows are easier to tap
  constexpr int MAX_ROWS = 10;
  constexpr int RSSI_X   = 280;
  constexpr int BARS_X   = 340;
  constexpr int MAX_SSID_CHARS = 18;

  constexpr int DOT_X    = Cfg::SCREEN_W - MARGIN - 14;  // "scanning" indicator
  constexpr int DOT_Y    = TITLE_Y + 10;
  constexpr int DOT_SIZE = 12;

  // Result-screen triangle
  constexpr int TRI_CX     = 206;   // even, close to the centre (410 / 2 = 205)
  constexpr int TRI_TOP    = 100;
  constexpr int TRI_BOTTOM = 280;
  constexpr int TRI_HALF_W = 100;   // half the width of the flat side
}

// =====================================================================
//  Triangle animation
// =====================================================================
namespace Anim {
  constexpr int           REVEAL_ROWS_PER_STEP = 4;    // rows drawn per frame
  constexpr unsigned long REVEAL_STEP_MS       = 15;   // ~0.7 s total reveal
  constexpr unsigned long PULSE_PERIOD_MS      = 2000; // white -> dark grey -> white
  constexpr uint8_t       PULSE_DARK           = 60;   // darkest grey (0-255)
}

// =====================================================================
//  Colors (RGB565)
// =====================================================================
namespace Color {
  constexpr uint16_t BLACK   = 0x0000;
  constexpr uint16_t WHITE   = 0xFFFF;
  constexpr uint16_t GREY    = 0x7BEF;
  constexpr uint16_t ROW_BG  = 0x18E3;  // highlight behind a tapped row
  constexpr uint16_t CYAN    = 0x07FF;
  constexpr uint16_t GREEN   = 0x07E0;
  constexpr uint16_t YELLOW  = 0xFFE0;
  constexpr uint16_t RED     = 0xF800;
}

// =====================================================================
//  Types and global state
// =====================================================================
struct WifiNet {
  String ssid;
  int    rssi;
  bool   open;
  bool   hidden;
};

struct Touch {
  bool down;    // finger is on the screen
  bool tapped;  // finger just landed this loop
  int  x, y;
};

enum class Screen { List, Result };

Arduino_DataBus *bus = new Arduino_ESP32QSPI(
    Pin::LCD_CS, Pin::LCD_SCLK,
    Pin::LCD_SDIO0, Pin::LCD_SDIO1, Pin::LCD_SDIO2, Pin::LCD_SDIO3);

Arduino_CO5300 *gfx = new Arduino_CO5300(
    bus, Pin::LCD_RESET, 0 /* rotation */,
    Cfg::SCREEN_W, Cfg::SCREEN_H, Cfg::COL_OFFSET, 0, 0, 0);

WifiNet       networks[Layout::MAX_ROWS];  // latest scan results
int           networkCount = 0;
WifiNet       shown[Layout::MAX_ROWS];     // what's currently drawn on screen
int           shownCount   = 0;

Screen        screen       = Screen::List;
bool          scanning     = false;
unsigned long lastScanEnd  = 0;
char          lastStatus[32] = "Starting...";

bool          pulsing      = false;  // triangle colour cycling on the success screen
unsigned long pulseStart   = 0;
int           lastPulseLvl = -1;

// =====================================================================
//  Small helpers
// =====================================================================
uint16_t rssiColor(int rssi) {
  if (rssi > -60) return Color::GREEN;   // strong
  if (rssi > -75) return Color::YELLOW;  // okay
  return Color::RED;                     // weak
}

int signalLevel(int rssi) {              // 1-4 bars
  if (rssi > -55) return 4;
  if (rssi > -65) return 3;
  if (rssi > -75) return 2;
  return 1;
}

String truncate(const String &s, unsigned int maxChars) {
  if (s.length() <= maxChars) return s;
  return s.substring(0, maxChars - 1) + "~";
}

bool sameNetwork(const WifiNet &a, const WifiNet &b) {
  return a.ssid == b.ssid && a.rssi == b.rssi && a.open == b.open;
}

const char *failReason(wl_status_t s) {
  switch (s) {
    case WL_NO_SSID_AVAIL:  return "Network not found";
    case WL_CONNECT_FAILED: return "Wrong password?";
    default:                return "Timed out";
  }
}

// Grey level 0-255 -> RGB565 (equal red, green, blue)
uint16_t grey565(uint8_t v) {
  return ((v >> 3) << 11) | ((v >> 2) << 5) | (v >> 3);
}

// =====================================================================
//  Touch (FT3168, read registers directly)
// =====================================================================
void touchInit() {
  pinMode(Pin::TP_RST, OUTPUT);
  digitalWrite(Pin::TP_RST, LOW);  delay(10);   // reset pulse
  digitalWrite(Pin::TP_RST, HIGH); delay(200);  // let the chip boot
  Wire.begin(Pin::TP_SDA, Pin::TP_SCL);
  Wire.setClock(400000);
}

// Register 0x02 = number of touches, 0x03-0x06 = X/Y high and low bytes
bool touchReadRaw(int &x, int &y) {
  Wire.beginTransmission(Cfg::TOUCH_ADDR);
  Wire.write(0x02);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(Cfg::TOUCH_ADDR, (uint8_t)5) != 5) return false;

  uint8_t points = Wire.read() & 0x0F;
  uint8_t xh = Wire.read(), xl = Wire.read();
  uint8_t yh = Wire.read(), yl = Wire.read();
  if (points == 0) return false;

  x = ((xh & 0x0F) << 8) | xl;   // top 4 bits are flags, not position
  y = ((yh & 0x0F) << 8) | yl;
  return true;
}

Touch touchPoll() {
  static bool wasDown = false;
  Touch t{};
  t.down   = touchReadRaw(t.x, t.y);
  t.tapped = t.down && !wasDown;  // react once per tap, not while held
  wasDown  = t.down;
  if (t.tapped) Serial.printf("Tap at x=%d y=%d\n", t.x, t.y);
  return t;
}

// =====================================================================
//  Drawing: basics (everything is 2-pixel aligned for the CO5300)
// =====================================================================
// Filled rectangle, expanded outward so it starts on even pixels and has
// even width and height. Use this instead of fillRect/drawLine/circles.
void rect2(int x, int y, int w, int h, uint16_t c) {
  if (w <= 0 || h <= 0) return;
  int x0 = x & ~1,             y0 = y & ~1;
  int x1 = (x + w + 1) & ~1,   y1 = (y + h + 1) & ~1;
  gfx->fillRect(x0, y0, x1 - x0, y1 - y0, c);
}

// Text: size must be even (2 or 4) and the position is rounded to even,
// so each character is drawn as 2x2 (or 4x4) aligned blocks.
void uiPrint(const String &text, int x, int y, int size, uint16_t color) {
  gfx->setTextSize(size);
  gfx->setTextColor(color);
  gfx->setCursor(x & ~1, y & ~1);
  gfx->print(text);
}

void uiPrintCentered(const String &text, int y, int size, uint16_t color) {
  int w = text.length() * 6 * size;  // built-in font: 6 px per char x size
  uiPrint(text, (Cfg::SCREEN_W - w) / 2, y, size, color);
}

// =====================================================================
//  Drawing: list screen
// =====================================================================
// Small square in the corner while a scan runs, instead of rewriting the header
void uiScanDot(bool on) {
  using namespace Layout;
  rect2(DOT_X, DOT_Y, DOT_SIZE, DOT_SIZE, on ? Color::CYAN : Color::BLACK);
}

// Only the status line is redrawn, so the title doesn't flicker
void uiStatus(const char *status) {
  if (status != lastStatus) strncpy(lastStatus, status, sizeof(lastStatus) - 1);
  rect2(0, Layout::STATUS_Y - 2, Cfg::SCREEN_W, 20, Color::BLACK);
  uiPrint(lastStatus, Layout::MARGIN, Layout::STATUS_Y, 2, Color::WHITE);
}

void uiHeader() {
  rect2(0, 0, Cfg::SCREEN_W, Layout::LIST_Y - 6, Color::BLACK);
  uiPrint("Wi-Fi Scan",            Layout::MARGIN, Layout::TITLE_Y, 4, Color::CYAN);
  uiPrint("Tap a network to join", Layout::MARGIN, Layout::HINT_Y,  2, Color::GREY);
  uiStatus(lastStatus);
  uiScanDot(scanning);
}

void uiBars(int x, int y, int rssi) {
  int level = signalLevel(rssi);
  for (int b = 0; b < 4; b++) {
    int h = 6 + b * 6;  // each bar taller than the last (even heights)
    rect2(x + b * 10, y + 24 - h, 6, h,
          (b < level) ? rssiColor(rssi) : Color::GREY);
  }
}

void uiClearRow(int i) {
  rect2(0, Layout::LIST_Y + i * Layout::ROW_H, Cfg::SCREEN_W, Layout::ROW_H, Color::BLACK);
}

void uiRow(int i, bool highlight) {
  const WifiNet &n = networks[i];
  int top = Layout::LIST_Y + i * Layout::ROW_H;

  rect2(Layout::MARGIN - 8, top, Cfg::SCREEN_W - 2 * (Layout::MARGIN - 8), Layout::ROW_H - 4,
        highlight ? Color::ROW_BG : Color::BLACK);

  uiPrint(truncate(n.ssid, Layout::MAX_SSID_CHARS), Layout::MARGIN, top + 8, 2,
          n.hidden ? Color::GREY : (n.open ? Color::YELLOW : Color::WHITE));
  uiPrint(String(n.rssi), Layout::RSSI_X, top + 8, 2, rssiColor(n.rssi));
  uiBars(Layout::BARS_X, top + 2, n.rssi);
}

// Redraw only the rows that actually changed since last time
void uiListUpdate() {
  int rows = max(networkCount, shownCount);
  for (int i = 0; i < rows; i++) {
    if (i >= networkCount) uiClearRow(i);
    else if (i >= shownCount || !sameNetwork(networks[i], shown[i])) uiRow(i, false);
  }
  for (int i = 0; i < networkCount; i++) shown[i] = networks[i];
  shownCount = networkCount;
}

// Full redraw, used when coming back from another screen
void uiListFull() {
  gfx->fillScreen(Color::BLACK);
  uiHeader();
  shownCount = 0;
  uiListUpdate();
}

// =====================================================================
//  Drawing: triangles (built from 2-pixel-tall horizontal bands)
// =====================================================================
// One 2-px band of the triangle at height y.
// pointUp = true:  point at the top, flat side at the bottom
// pointUp = false: flat side at the top, point at the bottom (inverted)
void uiTriangleBand(int y, bool pointUp, uint16_t c) {
  using namespace Layout;
  float t = pointUp ? float(y - TRI_TOP) / (TRI_BOTTOM - TRI_TOP)      // 0 at point
                    : float(TRI_BOTTOM - y) / (TRI_BOTTOM - TRI_TOP);  // 0 at point
  int half = TRI_HALF_W * t;
  if (half < 1) half = 1;  // keep a tiny tip visible
  rect2(TRI_CX - half, y, 2 * half, 2, c);
}

// Draw the triangle starting at the bottom and building up to the top
void uiTriangleReveal(bool pointUp, uint16_t c) {
  using namespace Layout;
  int band = 0;
  for (int y = TRI_BOTTOM - 2; y >= TRI_TOP; y -= 2) {
    uiTriangleBand(y, pointUp, c);
    band += 2;
    if (band % Anim::REVEAL_ROWS_PER_STEP == 0) delay(Anim::REVEAL_STEP_MS);
  }
}

// Recolour the whole triangle in one go
void uiTriangleFill(bool pointUp, uint16_t c) {
  using namespace Layout;
  for (int y = TRI_TOP; y < TRI_BOTTOM; y += 2) uiTriangleBand(y, pointUp, c);
}

// Smoothly cycle the success triangle white -> dark grey -> white.
// Non-blocking: called every loop while the success screen is up.
void uiTrianglePulse() {
  unsigned long t = (millis() - pulseStart) % Anim::PULSE_PERIOD_MS;
  float phase = 0.5f + 0.5f * cosf(2.0f * PI * t / Anim::PULSE_PERIOD_MS);  // 1 -> 0 -> 1
  int level = Anim::PULSE_DARK + (255 - Anim::PULSE_DARK) * phase;

  if (level == lastPulseLvl) return;  // nothing changed, skip the redraw
  lastPulseLvl = level;
  uiTriangleFill(false, grey565(level));
}

// =====================================================================
//  Drawing: connect / result screens
// =====================================================================
void uiConnecting(const WifiNet &n) {
  gfx->fillScreen(Color::BLACK);
  uiPrintCentered("Connecting to",      200, 2, Color::GREY);
  uiPrintCentered(truncate(n.ssid, 30), 230, 2, Color::CYAN);
}

void uiConnectingDots(int frame) {
  rect2(0, 280, Cfg::SCREEN_W, 30, Color::BLACK);
  for (int i = 0; i < 3; i++) {
    uint16_t c = (i == frame % 3) ? Color::WHITE : Color::GREY;
    rect2(Layout::TRI_CX - 44 + i * 30, 288, 14, 14, c);
  }
}

// Success: black background, white inverted triangle, then it starts cycling
void uiConnected(const WifiNet &n) {
  gfx->fillScreen(Color::BLACK);
  uiTriangleReveal(false, Color::WHITE);

  uiPrintCentered("Connected",                 320, 4, Color::WHITE);
  uiPrintCentered(truncate(n.ssid, 30),        368, 2, Color::CYAN);
  uiPrintCentered(WiFi.localIP().toString(),   398, 2, Color::WHITE);
  uiPrintCentered("Tap to go back",            450, 2, Color::GREY);
}

// Failure: fully white background, black triangle pointing up
void uiFailed(const WifiNet &n, const char *reason) {
  gfx->fillScreen(Color::WHITE);
  uiTriangleReveal(true, Color::BLACK);

  uiPrintCentered("Couldn't connect",          320, 4, Color::BLACK);
  uiPrintCentered(truncate(n.ssid, 30),        368, 2, Color::BLACK);
  uiPrintCentered(reason,                      398, 2, Color::RED);
  uiPrintCentered("Tap to go back",            450, 2, Color::GREY);
}

// =====================================================================
//  Wi-Fi scanning (runs in the background so touch stays responsive)
// =====================================================================
void scanStart() {
  WiFi.scanNetworks(true);  // true = async, returns immediately
  scanning = true;
  if (screen == Screen::List) uiScanDot(true);
}

// Let a running scan finish before connecting (the radio can't do both)
void scanStop() {
  while (WiFi.scanComplete() == WIFI_SCAN_RUNNING) delay(50);
  WiFi.scanDelete();
  scanning    = false;
  lastScanEnd = millis();
}

// Copy results, sort strongest first, keep the top MAX_ROWS
void scanCollect(int n) {
  static WifiNet all[Cfg::MAX_RESULTS];
  int total = min(n, Cfg::MAX_RESULTS);

  for (int i = 0; i < total; i++) {
    String ssid = WiFi.SSID(i);
    bool hidden = ssid.length() == 0;
    all[i] = { hidden ? "(hidden)" : ssid,
               (int)WiFi.RSSI(i),
               WiFi.encryptionType(i) == WIFI_AUTH_OPEN,
               hidden };
  }
  WiFi.scanDelete();  // free the driver's copy

  std::sort(all, all + total,
            [](const WifiNet &a, const WifiNet &b) { return a.rssi > b.rssi; });

  networkCount = min(total, Layout::MAX_ROWS);
  for (int i = 0; i < networkCount; i++) networks[i] = all[i];
}

void scanUpdate() {
  if (!scanning) {
    if (millis() - lastScanEnd >= Cfg::SCAN_INTERVAL_MS) scanStart();
    return;
  }

  int n = WiFi.scanComplete();
  if (n == WIFI_SCAN_RUNNING) return;  // still going, check again next loop

  scanning    = false;
  lastScanEnd = millis();
  uiScanDot(false);

  if (n < 0) {
    uiStatus("Scan failed");
    return;
  }

  scanCollect(n);

  char status[32];
  snprintf(status, sizeof(status), "%d networks found", n);
  if (strcmp(status, lastStatus) != 0) uiStatus(status);
  uiListUpdate();
}

// =====================================================================
//  Wi-Fi connecting
// =====================================================================
// Returns WL_CONNECTED on success, otherwise the last status seen
wl_status_t wifiConnect(const WifiNet &n) {
  uiConnecting(n);
  scanStop();

  WiFi.disconnect();
  delay(100);
  if (n.open) WiFi.begin(n.ssid.c_str());
  else        WiFi.begin(n.ssid.c_str(), Cfg::WIFI_PASSWORD);

  unsigned long start = millis();
  int frame = 0;
  wl_status_t s;
  while ((s = WiFi.status()) != WL_CONNECTED) {
    if (millis() - start > Cfg::CONNECT_TIMEOUT_MS ||
        s == WL_CONNECT_FAILED || s == WL_NO_SSID_AVAIL) {
      WiFi.disconnect();  // stop retrying so scanning works again
      return s;
    }
    uiConnectingDots(frame++);
    delay(300);
  }
  return WL_CONNECTED;
}

// =====================================================================
//  Screens
// =====================================================================
int rowAt(int y) {
  if (y < Layout::LIST_Y) return -1;
  int row = (y - Layout::LIST_Y) / Layout::ROW_H;
  return (row < networkCount) ? row : -1;
}

void handleListScreen(const Touch &t) {
  scanUpdate();
  if (!t.tapped) return;

  int row = rowAt(t.y);
  if (row < 0 || networks[row].hidden) return;  // can't join a hidden network by tap

  uiRow(row, true);  // flash the tapped row
  delay(Cfg::ROW_FLASH_MS);

  WifiNet target = networks[row];  // copy: the list may change while connecting
  Serial.printf("Connecting to %s...\n", target.ssid.c_str());
  screen = Screen::Result;

  wl_status_t s = wifiConnect(target);
  if (s == WL_CONNECTED) {
    Serial.printf("Connected, IP: %s\n", WiFi.localIP().toString().c_str());
    uiConnected(target);
    pulsing      = true;   // start the white <-> dark grey cycle
    pulseStart   = millis();
    lastPulseLvl = -1;
  } else {
    Serial.printf("Failed: %s (status %d)\n", failReason(s), s);
    uiFailed(target, failReason(s));
    pulsing = false;
  }
}

void handleResultScreen(const Touch &t) {
  if (pulsing) uiTrianglePulse();
  if (!t.tapped) return;

  pulsing = false;
  screen  = Screen::List;
  uiListFull();
}

// =====================================================================
//  Main
// =====================================================================
void setup() {
  Serial.begin(115200);
  delay(1500);  // give USB serial time to connect

  if (!gfx->begin()) Serial.println("Display init failed!");
  gfx->setBrightness(Cfg::BRIGHTNESS);

  touchInit();

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  uiListFull();
  scanStart();  // first scan right away
}

void loop() {
  Touch t = touchPoll();

  switch (screen) {
    case Screen::List:   handleListScreen(t);   break;
    case Screen::Result: handleResultScreen(t); break;
  }

  delay(Cfg::LOOP_DELAY_MS);
}
