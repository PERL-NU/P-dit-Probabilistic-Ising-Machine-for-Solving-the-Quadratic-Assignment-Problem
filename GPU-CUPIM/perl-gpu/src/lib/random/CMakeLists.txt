add_library(perl_random)
target_sources(perl_random
    PRIVATE
        random.cu
    PUBLIC
        FILE_SET HEADERS FILES random.cuh
)
target_link_libraries(perl_random
    mathdx::curanddx
)
