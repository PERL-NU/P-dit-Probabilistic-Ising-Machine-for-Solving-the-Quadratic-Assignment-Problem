add_executable(qap_gpu)
target_sources(qap_gpu PRIVATE qap_gpu.cu)
target_link_libraries(qap_gpu
    perl_qap
    perl_logger
)

add_executable(qap_cpu)
target_sources(qap_cpu PRIVATE qap_cpu.cpp)
target_link_libraries(qap_cpu)
