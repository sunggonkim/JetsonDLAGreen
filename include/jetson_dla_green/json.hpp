#pragma once

#include <iomanip>
#include <ostream>
#include <string_view>

namespace jdg {

inline void write_json_string(std::ostream& output, std::string_view value) {
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          const auto flags = output.flags();
          const auto fill = output.fill();
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character);
          output.flags(flags);
          output.fill(fill);
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
}

}  // namespace jdg
