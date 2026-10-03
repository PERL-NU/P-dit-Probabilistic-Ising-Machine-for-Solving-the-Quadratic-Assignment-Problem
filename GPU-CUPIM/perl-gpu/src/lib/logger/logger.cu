#include "logger.cuh"

#include <fstream>
#include <iostream>

//=====================================================================================
// Logger Class
//=====================================================================================

Logger::Logger(bool echo, std::optional<std::string> target)
: echo(echo), target(target)
{
};

Logger::~Logger() {
    if (this->stream.is_open()) {
        this->stream.close();
    }
};

void Logger::set_target(std::string new_target) {
    if (this->target.has_value()) {
        if (this->target.value() == new_target) {
            return;
        }
        this->stream.close();
    }
}

void Logger::write(std::string msg) {
    if (!this->target.has_value()) {
        std::cout << msg;
        return;
    }
    if (!this->stream.is_open()) {
        this->stream.open(this->target.value(), std::ios::app);
    }
    if (this->echo) {
        std::cout << msg;
    }
    this->stream << msg;
    this->stream.close();
}

//=====================================================================================
// Namespace Functions
//=====================================================================================

static Logger *_global_logger;

void logger::set_global(Logger *const logger) {
    _global_logger = logger;
}

void logger::write(const std::string msg) {
    if (_global_logger == nullptr) {
        std::cout << msg;
        return;
    }
    _global_logger->write(msg);
}
