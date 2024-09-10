#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

// Log levels
enum LogLevel
{
    INFO,
    WARNING,
    ERROR,
    FATAL
};

// Log class
class SimpleLog
{
public:
    SimpleLog(LogLevel level) : level_(level), condition_met_(true) {}

    // Conditional constructor for LOG_IF
    SimpleLog(LogLevel level, bool condition_met) : level_(level), condition_met_(condition_met) {}

    ~SimpleLog() noexcept(false)
    {
        if (!condition_met_)
        {
            return;  // Don't log anything if the condition was not met.
        }
        if (level_ == ERROR || level_ == FATAL)
        {
            std::cerr << (level_ == FATAL ? "FATAL: " : "ERROR: ") << stream_.str() << std::endl;
        }
        else if (level_ == WARNING)
        {
            std::cerr << "WARNING: " << stream_.str() << std::endl;
        }
        else
        {
            std::cout << "INFO: " << stream_.str() << std::endl;
        }

        if (level_ == FATAL)
        {
            throw std::runtime_error("Terminating due to fatal error.");
        }
    }

    template <typename T>
    SimpleLog& operator<<(const T& msg)
    {
        if (condition_met_)
        {
            stream_ << msg;
        }
        return *this;
    }

private:
    LogLevel           level_;
    bool               condition_met_;
    std::ostringstream stream_;
};

// Log conditionally
#define LOG_IF(level, condition) SimpleLog(level, (condition))

// Simple CHECK macro
#define FCHECK(condition) LOG_IF(FATAL, !(condition)) << "Check failed: " #condition " "

// CHECK_EQ and other checks
#define FCHECK_EQ(val1, val2)                                                                    \
    if ((val1) != (val2))                                                                        \
    SimpleLog(FATAL) << "Check failed: " << #val1 << " == " << #val2 << " (" << (val1) << " vs " \
                     << (val2) << ")"

// CHECK_LT and other checks
#define FCHECK_LT(val1, val2)                                                                   \
    if ((val1) >= (val2))                                                                       \
    SimpleLog(FATAL) << "Check failed: " << #val1 << " < " << #val2 << " (" << (val1) << " vs " \
                     << (val2) << ")"

// CHECK_LE and other checks
#define FCHECK_LE(val1, val2)                                                                    \
    if ((val1) > (val2))                                                                         \
    SimpleLog(FATAL) << "Check failed: " << #val1 << " <= " << #val2 << " (" << (val1) << " vs " \
                     << (val2) << ")"

// CHECK_GE and other checks
#define FCHECK_GE(val1, val2)                                                                    \
    if ((val1) < (val2))                                                                         \
    SimpleLog(FATAL) << "Check failed: " << #val1 << " >= " << #val2 << " (" << (val1) << " vs " \
                     << (val2) << ")"

// CHECK_GT and other checks
#define FCHECK_GT(val1, val2)                                                                    \
    if ((val1) <= (val2))                                                                        \
    SimpleLog(FATAL) << "Check failed: " << #val1 << " >= " << #val2 << " (" << (val1) << " vs " \
                     << (val2) << ")"

// Logging macros for different levels
#define LOG(level) SimpleLog(level)