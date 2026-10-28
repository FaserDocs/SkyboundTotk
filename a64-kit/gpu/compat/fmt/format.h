// The {fmt} API subset suyu's shader_recompiler uses, on C++20 std::format (no external {fmt} needed).
// Types given a fmt::formatter specialization are formattable by std::format too (the bridge at the end).
#pragma once
#include <format>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace fmt {
using std::format_context;
using std::format_parse_context;

template <typename T, typename Char = char, typename Enable = void>
struct formatter;   // specialized by the code that uses it

template <>
struct formatter<std::string_view> : std::formatter<std::string_view> {};

struct runtime_format_string {
    std::string_view str;
};
inline runtime_format_string runtime(std::string_view s) { return {s}; }

template <typename... Args>
std::string format(std::format_string<Args...> f, Args&&... args) {
    return std::format(f, std::forward<Args>(args)...);
}
template <typename... Args>
std::string format(runtime_format_string f, Args&&... args) {
    return std::vformat(f.str, std::make_format_args(args...));
}
template <typename Out, typename... Args>
Out format_to(Out out, std::format_string<Args...> f, Args&&... args) {
    return std::format_to(std::move(out), f, std::forward<Args>(args)...);
}
template <typename Out, typename... Args>
Out format_to(Out out, runtime_format_string f, Args&&... args) {
    return std::vformat_to(std::move(out), f.str, std::make_format_args(args...));
}
template <typename T>
std::string to_string(const T& value) {
    return std::format("{}", value);
}

namespace detail {
template <typename T>
concept has_fmt_formatter = requires { sizeof(::fmt::formatter<T>); };
template <typename T>
concept has_format_as = requires(const T& t) { format_as(t); };   // {fmt} 10's customization point (ADL)
}
} // namespace fmt

template <typename T>
    requires(fmt::detail::has_fmt_formatter<T> && !std::is_same_v<T, std::string_view>)
struct std::formatter<T, char> : ::fmt::formatter<T> {};

// {fmt} before 10 formatted enums as their underlying integer, and suyu relies on it; std::format does not.
template <typename T>
    requires(std::is_enum_v<T> && !fmt::detail::has_fmt_formatter<T> && !fmt::detail::has_format_as<T>)
struct std::formatter<T, char> : std::formatter<std::underlying_type_t<T>, char> {
    template <typename Ctx>
    auto format(T v, Ctx& ctx) const {
        return std::formatter<std::underlying_type_t<T>, char>::format(static_cast<std::underlying_type_t<T>>(v), ctx);
    }
};

template <typename T>
    requires(fmt::detail::has_format_as<T> && !fmt::detail::has_fmt_formatter<T>)
struct std::formatter<T, char> : std::formatter<std::decay_t<decltype(format_as(std::declval<const T&>()))>, char> {
    template <typename Ctx>
    auto format(const T& v, Ctx& ctx) const {
        using Base = std::formatter<std::decay_t<decltype(format_as(std::declval<const T&>()))>, char>;
        return Base::format(format_as(v), ctx);
    }
};
