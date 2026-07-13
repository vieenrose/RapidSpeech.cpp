// rs_json.h — a compact, self-contained JSON value + parser + serializer.
//
// Just enough for the server's OpenAI request/response bodies and the MCP
// JSON-RPC messages: objects, arrays, strings (with \uXXXX + surrogate pairs),
// numbers, booleans, null. Not a general-purpose library — no comments, no
// streaming — but correct for well-formed input from OpenAI / MCP clients.
//
// Usage:
//   rsjson::Value v = rsjson::parse(body);          // throws rsjson::Error
//   std::string t = v["input"].as_string();         // "" if missing/!string
//   rsjson::Value o = rsjson::Value::object();
//   o["text"] = "hello";                             // implicit from const char*
//   std::string out = o.dump();

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsjson {

struct Error : std::runtime_error {
    explicit Error(const std::string &m) : std::runtime_error(m) {}
};

class Value;
using Array = std::vector<Value>;
// ordered map keeps object key order stable for readable output
using Object = std::vector<std::pair<std::string, Value>>;

enum class Type { Null, Bool, Number, String, Array, Object };

class Value {
public:
    Value() : type_(Type::Null) {}
    Value(std::nullptr_t) : type_(Type::Null) {}
    Value(bool b) : type_(Type::Bool), bool_(b) {}
    Value(int n) : type_(Type::Number), num_(n) {}
    Value(int64_t n) : type_(Type::Number), num_((double)n) {}
    Value(size_t n) : type_(Type::Number), num_((double)n) {}
    Value(double n) : type_(Type::Number), num_(n) {}
    Value(const char *s) : type_(Type::String), str_(s) {}
    Value(const std::string &s) : type_(Type::String), str_(s) {}
    Value(std::string &&s) : type_(Type::String), str_(std::move(s)) {}

    static Value object() { Value v; v.type_ = Type::Object; return v; }
    static Value array()  { Value v; v.type_ = Type::Array;  return v; }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_object() const { return type_ == Type::Object; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_string() const { return type_ == Type::String; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_bool() const { return type_ == Type::Bool; }

    // Object access. Reading a missing key on an object returns a Null value
    // (const overload returns a static Null); writing auto-creates.
    Value &operator[](const std::string &key) {
        if (type_ != Type::Object) { type_ = Type::Object; obj_.clear(); }
        for (auto &kv : obj_) if (kv.first == key) return kv.second;
        obj_.emplace_back(key, Value());
        return obj_.back().second;
    }
    const Value &operator[](const std::string &key) const {
        static const Value null_v;
        if (type_ == Type::Object)
            for (auto &kv : obj_) if (kv.first == key) return kv.second;
        return null_v;
    }
    bool contains(const std::string &key) const {
        if (type_ != Type::Object) return false;
        for (auto &kv : obj_) if (kv.first == key) return true;
        return false;
    }

    // Array access / build.
    void push_back(Value v) {
        if (type_ != Type::Array) { type_ = Type::Array; arr_.clear(); }
        arr_.push_back(std::move(v));
    }
    const Array &items() const { return arr_; }
    Array &items() { return arr_; }
    const Object &members() const { return obj_; }

    // Typed extraction with defaults (never throws).
    std::string as_string(const std::string &d = "") const {
        return type_ == Type::String ? str_ : d;
    }
    double as_number(double d = 0) const {
        return type_ == Type::Number ? num_ : (type_ == Type::Bool ? (bool_ ? 1 : 0) : d);
    }
    int as_int(int d = 0) const { return type_ == Type::Number ? (int)num_ : d; }
    bool as_bool(bool d = false) const {
        return type_ == Type::Bool ? bool_ : (type_ == Type::Number ? num_ != 0 : d);
    }

    std::string dump(int indent = 0) const {
        std::ostringstream os;
        write(os, indent, 0);
        return os.str();
    }

private:
    Type type_;
    bool bool_ = false;
    double num_ = 0;
    std::string str_;
    Array arr_;
    Object obj_;

    friend Value parse(const std::string &);

    static void escape(std::ostream &os, const std::string &s) {
        os << '"';
        for (unsigned char c : s) {
            switch (c) {
                case '"':  os << "\\\""; break;
                case '\\': os << "\\\\"; break;
                case '\n': os << "\\n";  break;
                case '\r': os << "\\r";  break;
                case '\t': os << "\\t";  break;
                case '\b': os << "\\b";  break;
                case '\f': os << "\\f";  break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        os << buf;
                    } else {
                        os << (char)c; // pass UTF-8 bytes through unescaped
                    }
            }
        }
        os << '"';
    }

    void write(std::ostream &os, int indent, int depth) const {
        const bool pretty = indent > 0;
        const std::string pad = pretty ? std::string((size_t)indent * (depth + 1), ' ') : "";
        const std::string pad0 = pretty ? std::string((size_t)indent * depth, ' ') : "";
        const char *nl = pretty ? "\n" : "";
        switch (type_) {
            case Type::Null:   os << "null"; break;
            case Type::Bool:   os << (bool_ ? "true" : "false"); break;
            case Type::Number: {
                if (num_ == (int64_t)num_) os << (int64_t)num_;
                else { std::ostringstream t; t.precision(17); t << num_; os << t.str(); }
                break;
            }
            case Type::String: escape(os, str_); break;
            case Type::Array: {
                if (arr_.empty()) { os << "[]"; break; }
                os << '[' << nl;
                for (size_t i = 0; i < arr_.size(); ++i) {
                    os << pad; arr_[i].write(os, indent, depth + 1);
                    if (i + 1 < arr_.size()) os << ',';
                    os << nl;
                }
                os << pad0 << ']';
                break;
            }
            case Type::Object: {
                if (obj_.empty()) { os << "{}"; break; }
                os << '{' << nl;
                for (size_t i = 0; i < obj_.size(); ++i) {
                    os << pad; escape(os, obj_[i].first);
                    os << (pretty ? ": " : ":");
                    obj_[i].second.write(os, indent, depth + 1);
                    if (i + 1 < obj_.size()) os << ',';
                    os << nl;
                }
                os << pad0 << '}';
                break;
            }
        }
    }
};

// ── parser ──────────────────────────────────────────────────────────────
namespace detail {

struct Parser {
    const char *p, *end;
    explicit Parser(const std::string &s) : p(s.data()), end(s.data() + s.size()) {}

    [[noreturn]] void fail(const char *msg) { throw Error(std::string("JSON: ") + msg); }
    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    char peek() { ws(); return p < end ? *p : '\0'; }

    Value parse_value() {
        char c = peek();
        switch (c) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': return Value(parse_string());
            case 't': case 'f': return parse_bool();
            case 'n': expect_lit("null"); return Value(nullptr);
            default:  return parse_number();
        }
    }

    void expect_lit(const char *lit) {
        for (const char *q = lit; *q; ++q) {
            if (p >= end || *p != *q) fail("bad literal");
            ++p;
        }
    }
    Value parse_bool() {
        if (peek() == 't') { expect_lit("true"); return Value(true); }
        expect_lit("false"); return Value(false);
    }
    Value parse_number() {
        ws();
        const char *s = p;
        while (p < end && (*p == '-' || *p == '+' || *p == '.' || *p == 'e' ||
                           *p == 'E' || (*p >= '0' && *p <= '9'))) ++p;
        if (p == s) fail("expected value");
        return Value(std::stod(std::string(s, p)));
    }
    std::string parse_string() {
        ws();
        if (*p != '"') fail("expected string");
        ++p;
        std::string out;
        while (p < end && *p != '"') {
            char c = *p++;
            if (c == '\\') {
                if (p >= end) fail("bad escape");
                char e = *p++;
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': out += parse_unicode(); break;
                    default: fail("bad escape char");
                }
            } else {
                out += c;
            }
        }
        if (p >= end) fail("unterminated string");
        ++p; // closing quote
        return out;
    }
    unsigned hex4() {
        if (end - p < 4) fail("bad \\u");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex");
        }
        return v;
    }
    std::string parse_unicode() {
        unsigned cp = hex4();
        if (cp >= 0xD800 && cp <= 0xDBFF) { // high surrogate
            if (end - p >= 2 && p[0] == '\\' && p[1] == 'u') {
                p += 2;
                unsigned lo = hex4();
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            }
        }
        std::string u;
        if (cp < 0x80) u += (char)cp;
        else if (cp < 0x800) {
            u += (char)(0xC0 | (cp >> 6));
            u += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            u += (char)(0xE0 | (cp >> 12));
            u += (char)(0x80 | ((cp >> 6) & 0x3F));
            u += (char)(0x80 | (cp & 0x3F));
        } else {
            u += (char)(0xF0 | (cp >> 18));
            u += (char)(0x80 | ((cp >> 12) & 0x3F));
            u += (char)(0x80 | ((cp >> 6) & 0x3F));
            u += (char)(0x80 | (cp & 0x3F));
        }
        return u;
    }
    Value parse_array() {
        ++p; // [
        Value v = Value::array();
        if (peek() == ']') { ++p; return v; }
        for (;;) {
            v.push_back(parse_value());
            char c = peek();
            if (c == ',') { ++p; continue; }
            if (c == ']') { ++p; break; }
            fail("expected , or ]");
        }
        return v;
    }
    Value parse_object() {
        ++p; // {
        Value v = Value::object();
        if (peek() == '}') { ++p; return v; }
        for (;;) {
            if (peek() != '"') fail("expected object key");
            std::string k = parse_string();
            if (peek() != ':') fail("expected :");
            ++p;
            v[k] = parse_value();
            char c = peek();
            if (c == ',') { ++p; continue; }
            if (c == '}') { ++p; break; }
            fail("expected , or }");
        }
        return v;
    }
};

} // namespace detail

inline Value parse(const std::string &s) {
    detail::Parser pr(s);
    Value v = pr.parse_value();
    return v;
}

} // namespace rsjson
