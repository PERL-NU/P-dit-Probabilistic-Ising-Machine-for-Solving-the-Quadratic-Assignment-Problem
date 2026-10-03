#include "layout.cuh"

#include "cuda_runtime.h"

#include <fstream>
#include <stdexcept>
#include <stdint.h>
#include <string>

using Layout = qap::Layout;

void Layout::cu_sync(Layout &dst, const cudaMemcpyKind direction) {
    if (dst.problem_size != this->problem_size) {
        throw std::runtime_error("cannot sync with differing problem sizes");
    }
    cudaMemcpy(
        dst.position_of_facility,
        this->position_of_facility,
        this->problem_size * sizeof(Position),
        direction
    );
    cudaMemcpy(
        dst.facility_at_position,
        this->facility_at_position,
        this->problem_size * sizeof(Facility),
        direction
    );
    const size_t size_1d = sizeof(int32_t) * problem_size;
    const size_t size_2d = sizeof(int32_t) * problem_size * problem_size;
    cudaMemcpy(dst.flow_mat, this->flow_mat, size_2d, direction);
    cudaMemcpy(dst.flow_mat_transposed, this->flow_mat_transposed, size_2d, direction);
    cudaMemcpy(dst.dist_mat, this->dist_mat, size_2d, direction);
    cudaMemcpy(dst.dist_mat_diag, this->dist_mat_diag, size_1d, direction);
    cudaMemcpy(dst.dist_mat_transposed, this->dist_mat_transposed, size_2d, direction);
}

Layout Layout::cu_copy(const cudaMemcpyKind direction) {
    Layout new_layout = {
        .problem_size = this->problem_size,
    };
    const size_t size_1d = this->problem_size;
    const size_t size_2d = this->problem_size * this->problem_size;
    switch (direction) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyDeviceToHost:
            new_layout.on_device = false;
            new_layout.position_of_facility = new Position[size_1d];
            new_layout.facility_at_position = new Facility[size_1d];
            new_layout.flow_mat = new int32_t[size_2d];
            new_layout.flow_mat_transposed = new int32_t[size_2d];
            new_layout.dist_mat = new int32_t[size_2d];
            new_layout.dist_mat_diag = new int32_t[size_1d];
            new_layout.dist_mat_transposed = new int32_t[size_2d];
            break;
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToDevice:
            new_layout.on_device = true;
            cudaMalloc(&new_layout.position_of_facility, sizeof(Position) * size_1d);
            cudaMalloc(&new_layout.facility_at_position, sizeof(Facility) * size_1d);
            cudaMalloc(&new_layout.flow_mat, sizeof(int32_t) * size_2d);
            cudaMalloc(&new_layout.flow_mat_transposed, sizeof(int32_t) * size_2d);
            cudaMalloc(&new_layout.dist_mat, sizeof(int32_t) * size_2d);
            cudaMalloc(&new_layout.dist_mat_diag, sizeof(int32_t) * size_1d);
            cudaMalloc(&new_layout.dist_mat_transposed, sizeof(int32_t) * size_2d);
            break;
        default:
            throw("unrecognized cudaMemcpyKind");
    }
    this->cu_sync(new_layout, direction);
    return new_layout;
}

void Layout::cu_free() {
    if (this->on_device) {
        cudaFree(this->position_of_facility);
        cudaFree(this->facility_at_position);
        cudaFree(this->flow_mat);
        cudaFree(this->flow_mat_transposed);
        cudaFree(this->dist_mat);
        cudaFree(this->dist_mat_diag);
        cudaFree(this->dist_mat_transposed);
    } else {
        delete[] this->position_of_facility;
        delete[] this->facility_at_position;
        delete[] this->flow_mat;
        delete[] this->flow_mat_transposed;
        delete[] this->dist_mat;
        delete[] this->dist_mat_diag;
        delete[] this->dist_mat_transposed;
    }
}

Layout Layout::from_file(const std::string &filename) {
    std::ifstream input_file(filename);
    if (!input_file.is_open()) {
        throw std::runtime_error("failed to open file");
    }
    Layout new_layout = {
        .on_device = false,
    };

    // Problem size and distance/flow matrices come from file
    input_file >> new_layout.problem_size;
    const uint32_t ps = new_layout.problem_size;

    new_layout.dist_mat = new int32_t[ps * ps];
    new_layout.dist_mat_diag = new int32_t[ps];
    new_layout.dist_mat_transposed = new int32_t[ps * ps];
    new_layout.flow_mat = new int32_t[ps * ps];
    new_layout.flow_mat_transposed = new int32_t[ps * ps];
    new_layout.position_of_facility = new Position[ps];
    new_layout.facility_at_position = new Position[ps];

    for (int ii = 0; ii < ps; ++ii) {
        for (int jj = 0; jj < ps; ++jj) {
            input_file >> new_layout.flow_mat[ii * ps + jj];
            new_layout.flow_mat_transposed[ii + ps * jj] = new_layout.flow_mat[ii * ps + jj];
        }
    }
    for (int ii = 0; ii < ps; ++ii) {
        for (int jj = 0; jj < ps; ++jj) {
            input_file >> new_layout.dist_mat[ii * ps + jj];
            new_layout.dist_mat_transposed[ii + ps * jj] = new_layout.dist_mat[ii * ps + jj];
        }
    }
    for (int ii = 0; ii < ps; ++ii) {
        new_layout.dist_mat_diag[ii] = new_layout.dist_mat[ii * ps + ii];
    }
    input_file.close();

    // Initialize positions and factories in counting order
    for (int i = 0; i < new_layout.problem_size; ++i) {
        new_layout.facility_at_position[i] = i;
        new_layout.position_of_facility[i] = i;
    }
    return new_layout;
}
