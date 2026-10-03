add_library(perl_qap)

target_sources(perl_qap
    PRIVATE
        qap.cu
    PUBLIC
        FILE_SET HEADERS FILES qap.cuh
)
target_link_libraries(perl_qap
    mathdx::curanddx
)

target_sources(perl_qap
    PRIVATE
        qap.cu layout.cu
    PUBLIC
        FILE_SET HEADERS FILES qap.cuh
)
target_link_libraries(perl_qap
    mathdx::curanddx
    perl_random
)
