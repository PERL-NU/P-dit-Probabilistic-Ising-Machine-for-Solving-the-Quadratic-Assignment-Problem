#ifndef PERL_LOGGER_HPP
#define PERL_LOGGER_HPP

#include <fstream>
#include <optional>
#include <stdbool.h>
#include <string>

class Logger {
private:
    std::optional<std::string> target = std::nullopt;
    std::ofstream stream;
public:
    bool echo = false;

    Logger(bool echo=false, std::optional<std::string> target=std::nullopt);
    ~Logger();

    void set_target(std::string new_target);
    void write(std::string msg);
};

namespace logger {
    void set_global(Logger *const logger);
    void write(const std::string msg);
}

#endif  /* PERL_LOGGER_HPP */
