#ifndef PERL_QAP_LAYOUT
#define PERL_QAP_LAYOUT

#include "cuda_runtime.h"

#include <stdint.h>
#include <string>

namespace qap {
    using Position = uint32_t;
    using Facility = uint32_t;

    /**
     * @brief Specification for a facility location problem
     */
    struct Layout {

        /** Number of facilities/locations in the problem */
        uint32_t problem_size;

        /** Position lookup by facility */
        Position *position_of_facility;

        /** Facility lookup by position */
        Facility *facility_at_position;

        /**
         * @brief Flow between facilities
         *
         * flow_mat[a][b] represents the "flow of goods" from facility a to facility b.
         */
        int32_t *flow_mat;
        int32_t *flow_mat_transposed;

        /**
         * @brief Distance between positions
         *
         * dist_mat[i][j] represents the distance from position i to position j. This is
         * a symmetric matrix in the physical case, but may be asymmetrical for some
         * problems.
         */
        int32_t *dist_mat;
        int32_t *dist_mat_transposed;
        int32_t *dist_mat_diag;

        /** Flag if this structure is stored on device memory */
        bool on_device;

        // Methods
        inline void cu_sync(Layout &dst, const cudaMemcpyKind direction);
        inline Layout cu_copy(const cudaMemcpyKind direction);
        inline void cu_free();
        static inline Layout from_file(const std::string &filename);
    };
}

#endif  /* PERL_QAP_LAYOUT */
