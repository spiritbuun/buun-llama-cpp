#include "jinja/string.h"
#include "jinja/value.h"
#include "unicode.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace jinja {

static size_t utf8_char_len(const std::string & s, size_t pos) {
    if (static_cast<unsigned char>(s[pos]) < 0x80) {
        return 1;
    }
    const auto r = common_parse_utf8_codepoint(s, pos);
    return r.status == utf8_parse_result::SUCCESS ? r.bytes_consumed : 1;
}

bool utf8_is_ascii(const std::string & s) {
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return c < 0x80; });
}

size_t utf8_length(const std::string & s) {
    size_t n = 0;
    for (size_t pos = 0; pos < s.size(); pos += utf8_char_len(s, pos)) {
        ++n;
    }
    return n;
}

size_t utf8_offset(const std::string & s, size_t n_chars, size_t pos) {
    for (; n_chars > 0 && pos < s.size(); --n_chars) {
        pos += utf8_char_len(s, pos);
    }
    return pos;
}

std::vector<std::string> utf8_chars(const std::string & s) {
    std::vector<std::string> chars;
    chars.reserve(s.size());
    for (size_t pos = 0; pos < s.size();) {
        const size_t len = utf8_char_len(s, pos);
        chars.push_back(s.substr(pos, len));
        pos += len;
    }
    return chars;
}

//
// string_part
//

bool string_part::is_uppercase() const {
    for (char c : val) {
        if (std::islower(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

bool string_part::is_lowercase() const {
    for (char c : val) {
        if (std::isupper(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

//
// string
//

void string::mark_input() {
    for (auto & part : parts) {
        part.is_input = true;
    }
}

std::string string::str() const {
    if (parts.size() == 1) {
        return parts[0].val;
    }
    std::ostringstream oss;
    for (const auto & part : parts) {
        oss << part.val;
    }
    return oss.str();
}

size_t string::length() const {
    size_t len = 0;
    for (const auto & part : parts) {
        len += part.val.length();
    }
    return len;
}

void string::hash_update(hasher & hash) const noexcept {
    for (const auto & part : parts) {
        hash.update(part.val.data(), part.val.length());
    }
}

bool string::all_parts_are_input() const {
    for (const auto & part : parts) {
        if (!part.is_input) {
            return false;
        }
    }
    return true;
}

bool string::is_uppercase() const {
    for (const auto & part : parts) {
        if (!part.is_uppercase()) {
            return false;
        }
    }
    return true;
}

bool string::is_lowercase() const {
    for (const auto & part : parts) {
        if (!part.is_lowercase()) {
            return false;
        }
    }
    return true;
}

// mark this string as input if other has ALL parts as input
void string::mark_input_based_on(const string & other) {
    if (other.all_parts_are_input()) {
        for (auto & part : parts) {
            part.is_input = true;
        }
    }
}

string & string::append(const string & other) {
    for (const auto & part : other.parts) {
        parts.push_back(part);
    }
    return *this;
}

// in-place transformation

using transform_fn = std::function<std::string(const std::string&)>;
static string apply_transform(string & self, const transform_fn & fn) {
    for (auto & part : self.parts) {
        part.val = fn(part.val);
    }
    return self;
}

string string::uppercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::toupper);
        return res;
    });
}
string string::lowercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::tolower);
        return res;
    });
}
string string::capitalize() {
    return apply_transform(*this, [](const std::string & s) {
        if (s.empty()) return s;
        std::string res = s;
        res[0] = ::toupper(static_cast<unsigned char>(res[0]));
        std::transform(res.begin() + 1, res.end(), res.begin() + 1, ::tolower);
        return res;
    });
}
string string::titlecase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        bool capitalize_next = true;
        for (char &c : res) {
            if (isspace(static_cast<unsigned char>(c))) {
                capitalize_next = true;
            } else if (capitalize_next) {
                c = ::toupper(static_cast<unsigned char>(c));
                capitalize_next = false;
            } else {
                c = ::tolower(static_cast<unsigned char>(c));
            }
        }
        return res;
    });
}
string string::strip(bool left, bool right, std::optional<const std::string_view> chars) {
    static auto strip_part = [](const std::string & s, bool left, bool right, std::optional<const std::string_view> chars) -> std::string {
        size_t start = 0;
        size_t end = s.length();
        auto match_char = [&chars](unsigned char c) -> bool {
            return chars ? (*chars).find(c) != std::string::npos : isspace(c);
        };
        if (left) {
            while (start < end && match_char(static_cast<unsigned char>(s[start]))) {
                ++start;
            }
        }
        if (right) {
            while (end > start && match_char(static_cast<unsigned char>(s[end - 1]))) {
                --end;
            }
        }
        return s.substr(start, end - start);
    };
    if (parts.empty()) {
        return *this;
    }
    if (left) {
        for (size_t i = 0; i < parts.size(); ++i) {
            parts[i].val = strip_part(parts[i].val, true, false, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                --i;
                continue;
            } else {
                break;
            }
        }
    }
    if (right) {
        for (size_t i = parts.size(); i-- > 0;) {
            parts[i].val = strip_part(parts[i].val, false, true, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                continue;
            } else {
                break;
            }
        }
    }
    return *this;
}

} // namespace jinja
