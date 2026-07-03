#pragma once
#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <cstddef>

namespace propminer {

enum class JsonType { Null, Bool, Number, String, Array, Object };

class JsonValue {
public:
    using Object = std::vector<std::pair<std::string, JsonValue>>;
    using Array  = std::vector<JsonValue>;

    static JsonValue null_value();
    static JsonValue make_bool(bool v);
    static JsonValue make_number(double v);
    static JsonValue make_string(const std::string& v);
    static JsonValue make_array();
    static JsonValue make_object();

    JsonType type() const { return type_; }
    bool is_null()  const { return type_ == JsonType::Null; }
    bool is_bool()  const { return type_ == JsonType::Bool; }
    bool is_number() const { return type_ == JsonType::Number; }
    bool is_string() const { return type_ == JsonType::String; }
    bool is_array()  const { return type_ == JsonType::Array; }
    bool is_object() const { return type_ == JsonType::Object; }

    bool to_bool()  const;
    double to_number() const;
    int64_t to_int() const;
    uint32_t to_uint32() const;
    const std::string& to_string() const;
    const Array& to_array() const;
    const Object& to_object() const;

    // Mutators
    void push_back(const JsonValue& v);          // array
    void set(const std::string& key, const JsonValue& v); // object
    void set(const std::string& key, bool v);
    void set(const std::string& key, double v);
    void set(const std::string& key, int64_t v);
    void set(const std::string& key, uint32_t v);
    void set(const std::string& key, const std::string& v);

    // Accessors
    const JsonValue& operator[](const std::string& key) const;
    const JsonValue& operator[](size_t index) const;
    size_t size() const;

    // Serialize to JSON string
    std::string serialize() const;

    // Parse from string (returns null_value on failure)
    static JsonValue parse(const char* input, size_t len,
                           size_t* consumed = nullptr);
    static JsonValue parse(const std::string& input);

private:
    JsonType type_{JsonType::Null};
    std::variant<bool, double, std::string, Array, Object> data_;
};

} // namespace propminer