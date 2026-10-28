// Rexcomp replacement for suyu's common/logging.h: log lines go nowhere unless GPU_SHADER_LOG is defined.
#pragma once
#include <fmt/format.h>   // suyu code reaches fmt through this header
#ifdef GPU_SHADER_LOG
#include <cstdio>
#define LOG_GENERIC_(level, cls, ...) std::fprintf(stderr, "[shader %s] %s\n", level, fmt::format(__VA_ARGS__).c_str())
#else
#define LOG_GENERIC_(level, cls, ...) ((void)0)
#endif
#define LOG_TRACE(cls, ...) ((void)0)
#define LOG_DEBUG(cls, ...) ((void)0)
#define LOG_INFO(cls, ...) LOG_GENERIC_("info", cls, __VA_ARGS__)
#define LOG_WARNING(cls, ...) LOG_GENERIC_("warning", cls, __VA_ARGS__)
#define LOG_ERROR(cls, ...) LOG_GENERIC_("error", cls, __VA_ARGS__)
#define LOG_CRITICAL(cls, ...) LOG_GENERIC_("critical", cls, __VA_ARGS__)
