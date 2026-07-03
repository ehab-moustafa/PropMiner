#include "simple_json.h"
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

namespace propminer {

// ── Factory / accessors ──────────────────────────────────────────────────

JsonValue JsonValue::null_value() { return JsonValue(); }

JsonValue JsonValue::make_bool(bool v) {
    JsonValue j; j.type_ = JsonType::Bool; j.data_ = v; return j;
}
JsonValue JsonValue::make_number(double v) {
    JsonValue j; j.type_ = JsonType::Number; j.data_ = v; return j;
}
JsonValue JsonValue::make_string(const std::string& v) {
    JsonValue j; j.type_ = JsonType::String; j.data_ = v; return j;
}
JsonValue JsonValue::make_array() {
    JsonValue j; j.type_ = JsonType::Array; j.data_ = Array{}; return j;
}
JsonValue JsonValue::make_object() {
    JsonValue j; j.type_ = JsonType::Object; j.data_ = Object{}; return j;
}

bool JsonValue::to_bool() const { return std::get<bool>(data_); }
double JsonValue::to_number() const { return std::get<double>(data_); }
int64_t JsonValue::to_int() const { return static_cast<int64_t>(std::get<double>(data_)); }
uint32_t JsonValue::to_uint32() const { return static_cast<uint32_t>(std::get<double>(data_)); }
const std::string& JsonValue::to_string() const { return std::get<std::string>(data_); }
const JsonValue::Array& JsonValue::to_array() const { return std::get<Array>(data_); }
const JsonValue::Object& JsonValue::to_object() const { return std::get<Object>(data_); }

void JsonValue::push_back(const JsonValue& v) {
    if (type_ == JsonType::Array) std::get<Array>(data_).push_back(v);
}
void JsonValue::set(const std::string& key, const JsonValue& v) {
    if (type_ == JsonType::Object) std::get<Object>(data_).push_back({key, v});
}
void JsonValue::set(const std::string& key, bool v)       { set(key, make_bool(v)); }
void JsonValue::set(const std::string& key, double v)      { set(key, make_number(v)); }
void JsonValue::set(const std::string& key, int64_t v)     { set(key, make_number(static_cast<double>(v))); }
void JsonValue::set(const std::string& key, uint32_t v)    { set(key, make_number(static_cast<double>(v))); }
void JsonValue::set(const std::string& key, const std::string& v) { set(key, make_string(v)); }

const JsonValue& JsonValue::operator[](const std::string& key) const {
    static JsonValue s_null = null_value();
    if (type_ != JsonType::Object) return s_null;
    for (auto& p : std::get<Object>(data_))
        if (p.first == key) return p.second;
    return s_null;
}
const JsonValue& JsonValue::operator[](size_t index) const {
    static JsonValue s_null = null_value();
    if (type_ != JsonType::Array) return s_null;
    const auto& arr = std::get<Array>(data_);
    if (index >= arr.size()) return s_null;
    return arr[index];
}
size_t JsonValue::size() const {
    if (type_ == JsonType::Array)  return std::get<Array>(data_).size();
    if (type_ == JsonType::Object) return std::get<Object>(data_).size();
    return 0;
}

// ── Serialization ────────────────────────────────────────────────────────

static std::string escape_json_string(const std::string& s) {
    std::string out;
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    out.push_back('"');
    return out;
}

std::string JsonValue::serialize() const {
    switch (type_) {
        case JsonType::Null:   return "null";
        case JsonType::Bool:   return std::get<bool>(data_) ? "true" : "false";
        case JsonType::Number: {
            double v = std::get<double>(data_);
            if (v == std::floor(v) && std::abs(v) < 1e15)
                return std::to_string(static_cast<int64_t>(v));
            return std::to_string(v);
        }
        case JsonType::String: return escape_json_string(std::get<std::string>(data_));
        case JsonType::Array: {
            std::string out = "[";
            bool first = true;
            for (auto& e : std::get<Array>(data_)) {
                if (!first) out += ",";
                out += e.serialize();
                first = false;
            }
            out += "]";
            return out;
        }
        case JsonType::Object: {
            std::string out = "{";
            bool first = true;
            for (auto& p : std::get<Object>(data_)) {
                if (!first) out += ",";
                out += escape_json_string(p.first) + ":" + p.second.serialize();
                first = false;
            }
            out += "}";
            return out;
        }
    }
    return "null";
}

// ── Parser ───────────────────────────────────────────────────────────────

class JsonParser {
public:
    JsonParser(const char* input, size_t len)
        : input_(input), len_(len), pos_(0) {}

    JsonValue parse() {
        skip_ws();
        if (pos_ >= len_) return JsonValue::null_value();
        return parse_value();
    }

    size_t get_pos() const { return pos_; }

private:
    const char* input_;
    size_t len_;
    size_t pos_;

    char peek() const { return pos_ < len_ ? input_[pos_] : '\0'; }
    char next()       { return pos_ < len_ ? input_[pos_++] : '\0'; }
    void skip_ws()    { while (pos_ < len_ && (input_[pos_] == ' ' || input_[pos_] == '\t'
                     || input_[pos_] == '\n' || input_[pos_] == '\r')) pos_++; }

    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= len_) return JsonValue::null_value();
        char c = peek();
        if (c == '"')  return parse_string();
        if (c == '{')  return parse_object();
        if (c == '[')  return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n')  return parse_null();
        return parse_number();
    }

    JsonValue parse_string() {
        std::string s;
        next(); // skip opening "
        while (pos_ < len_ && peek() != '"') {
            if (peek() == '\\') {
                next();
                char esc = next();
                switch (esc) {
                    case '"':  s += '"'; break;
                    case '\\': s += '\\'; break;
                    case 'n':  s += '\n'; break;
                    case 'r':  s += '\r'; break;
                    case 't':  s += '\t'; break;
                    case '/':  s += '/'; break;
                    case 'b':  s += '\b'; break;
                    case 'f':  s += '\f'; break;
                    case 'u': {
                        // skip 4 hex digits
                        for (int i = 0; i < 4 && pos_ < len_; i++) next();
                        s += '?';
                        break;
                    }
                    default: s += esc; break;
                }
            } else {
                s += next();
            }
        }
        if (pos_ < len_) next(); // skip closing "
        return JsonValue::make_string(s);
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (peek() == '-') next();
        while (pos_ < len_ && std::isdigit(peek())) next();
        if (pos_ < len_ && peek() == '.') {
            next();
            while (pos_ < len_ && std::isdigit(peek())) next();
        }
        if (pos_ < len_ && (peek() == 'e' || peek() == 'E')) {
            next();
            if (pos_ < len_ && (peek() == '+' || peek() == '-')) next();
            while (pos_ < len_ && std::isdigit(peek())) next();
        }
        std::string num(input_ + start, pos_ - start);
        return JsonValue::make_number(std::stod(num));
    }

    JsonValue parse_bool() {
        if (input_[pos_] == 't') { pos_ += 4; return JsonValue::make_bool(true); }
        pos_ += 5;
        return JsonValue::make_bool(false);
    }

    JsonValue parse_null() {
        pos_ += 4;
        return JsonValue::null_value();
    }

    JsonValue parse_array() {
        auto j = JsonValue::make_array();
        next(); // skip [
        skip_ws();
        if (peek() == ']') { next(); return j; }
        while (true) {
            j.push_back(parse_value());
            skip_ws();
            if (peek() == ',') { next(); skip_ws(); }
            else break;
        }
        if (pos_ < len_) next(); // skip ]
        return j;
    }

    JsonValue parse_object() {
        auto j = JsonValue::make_object();
        next(); // skip {
        skip_ws();
        if (peek() == '}') { next(); return j; }
        while (true) {
            skip_ws();
            auto key = parse_string();
            skip_ws();
            if (pos_ < len_ && peek() == ':') next();
            j.set(key.to_string(), parse_value());
            skip_ws();
            if (peek() == ',') { next(); skip_ws(); }
            else break;
        }
        if (pos_ < len_) next(); // skip }
        return j;
    }
};

JsonValue JsonValue::parse(const char* input, size_t len, size_t* consumed) {
    JsonParser parser(input, len);
    auto result = parser.parse();
    if (consumed) *consumed = parser.get_pos();
    return result;
}

JsonValue JsonValue::parse(const std::string& input) {
    return parse(input.c_str(), input.size());
}

} // namespace propminer