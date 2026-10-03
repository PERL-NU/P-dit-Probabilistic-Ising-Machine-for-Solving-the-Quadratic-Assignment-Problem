add_library(perl_logger)

target_sources(perl_logger
    PRIVATE
        logger.cu
    PUBLIC
        FILE_SET HEADERS FILES logger.cuh
)
