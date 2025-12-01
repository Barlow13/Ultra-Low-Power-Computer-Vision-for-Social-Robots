#pragma once
#include <cstdint>
#include "hardware/i2c.h"

#ifndef SSD1306_WIDTH
#define SSD1306_WIDTH 128
#endif
#ifndef SSD1306_HEIGHT
#define SSD1306_HEIGHT 32
#endif
#ifndef SSD1306_I2C_ADDR
#define SSD1306_I2C_ADDR 0x3C
#endif

// Very small 5x7 font stored as 96*5 bytes (ASCII 0x20-0x7F)
extern const uint8_t g_font5x7[];

class SSD1306 {
public:
    bool init(i2c_inst_t* i2c, uint sda, uint scl, uint32_t speed_hz = 400000, uint8_t addr = SSD1306_I2C_ADDR);
    void clear();
    void drawChar(int x, int y, char c);
    void drawText(int x, int y, const char* text);
    void drawHLine(int x, int y, int w);
    void flush();
    void invert(bool inv);
    // Power control: turns panel on (true) or off (false) without losing RAM buffer.
    void power(bool on);
    bool isPowered() const { return _powered; }
private:
    void sendCommand(uint8_t cmd);
    void sendCommands(const uint8_t* cmds, int n);
    void sendBuffer();
    inline void setPixel(int x, int y, bool on);
private:
    i2c_inst_t* _i2c = nullptr;
    uint8_t _addr = SSD1306_I2C_ADDR;
    bool _inited = false;
    bool _inverted = false;
    bool _powered = true; // display starts ON after init
    uint8_t _buf[(SSD1306_WIDTH * SSD1306_HEIGHT)/8]{}; // 512 bytes
};
