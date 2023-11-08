/*
    Grouped GEMM for PyTorch
*/

// // // // // // // // // // // // // // // // // // // // // // // // 

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include <torch/extension.h>

namespace py = pybind11;

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous \n")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor \n")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// // // // // // // // // // // // // // // // // // // // // // // // 


auto getDeviceProps()
{
    int device_idx;
    cudaError_t status = cudaGetDevice(&device_idx);
    TORCH_CHECK(status == cudaSuccess, "cudaGetDevice() failed \n");

    cudaDeviceProp props;
    status = cudaGetDeviceProperties(&props, device_idx);
    TORCH_CHECK(status == cudaSuccess, "cudaGetDeviceProperties() failed \n");

    return props;
}

// // // // // // // // // // // // // // // // // // // // // // // // 


template <typename CutlassType> std::string type2str                  = "Unknown";
template <>                     std::string type2str<cutlass::half_t> = "Half";
template <>                     std::string type2str<float>           = "Float";

// // // // // // // // // // // // // // // // // // // // // // // // 

template <typename CutlassType> auto type2attr                  = at::ScalarType::Float;
template <>                     auto type2attr<cutlass::half_t> = at::ScalarType::Half;
template <>                     auto type2attr<float>           = at::ScalarType::Float;

// // // // // // // // // // // // // // // // // // // // // // // // 

template <typename CutlassType, int arch>
struct KernelConfig { using GemmKernel = void; };

// // // // 

template <>
struct KernelConfig<cutlass::half_t, 75>
{
    // cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align8
    using Gemm = cutlass::gemm::device::Gemm<
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<256, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>, 
            cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            2
        >;
};

template <>
struct KernelConfig<cutlass::half_t, 80>
{
    // cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nn_align8
    using Gemm = cutlass::gemm::device::Gemm<
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<256, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>, 
            cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            2
        >;

};

// // // // 
template <>
struct KernelConfig<float, 75>
{
    // cutlass_simt_sgemm_256x128_8x5_nn_align1
    using Gemm = cutlass::gemm::device::Gemm<
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            float, cutlass::arch::OpClassSimt, cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<256, 128, 8>,
            cutlass::gemm::GemmShape<64, 64, 8>, 
            cutlass::gemm::GemmShape<1, 1, 1>,
            cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            2
        >;
};

template <>
struct KernelConfig<float, 80>
{
    // cutlass_simt_sgemm_256x128_8x5_nn_align1
    using Gemm = cutlass::gemm::device::Gemm<
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            cutlass::half_t, cutlass::layout::RowMajor,
            float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<256, 128, 8>,
            cutlass::gemm::GemmShape<64, 64, 8>, 
            cutlass::gemm::GemmShape<1, 1, 1>,
            cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            2
        >;
};

// // // // // // // // // // // // // // // // // // // // // // // // 

/*
    Perform Matrix Multiplication
    D = [ alpha * A[i] x B[i] + beta * C]
*/
template <typename Gemm>
void GEMM_kernel(
        const torch::Tensor& matrix_A,
        const torch::Tensor& matrix_B,
        const torch::Tensor& matrix_C,
        const torch::Tensor& matrix_D,
        float alpha = 1.0, 
        float beta = 0.0
){
    /* some types */
    using ElementA = typename GemmKernel::ElementA;
    using ElementB = typename GemmKernel::ElementB;
    using ElementC = typename GemmKernel::ElementC;
    using LayoutA  = typename GemmKernel::LayoutA;
    using LayoutB  = typename GemmKernel::LayoutB;
    using LayoutC  = typename GemmKernel::LayoutC;
    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    using ElementComputeEpilogue = typename EpilogueOutputOp::ElementCompute;
    using MatrixCoord = typename LayoutC::TensorCoord;

    CHECK_INPUT(matrix_A);
    CHECK_INPUT(matrix_B);
    CHECK_INPUT(matrix_C);
    CHECK_INPUT(matrix_D);
    TORCH_CHECK(matrix_A.scalar_type() == type2attr<ElementA>, "input a wrong data type for matrix A \n");
    TORCH_CHECK(matrix_B.scalar_type() == type2attr<ElementB>, "input a wrong data type for matrix B \n");
    TORCH_CHECK(matrix_C.scalar_type() == type2attr<ElementC>, "input a wrong data type for matrix C \n");
    TORCH_CHECK(matrix_D.scalar_type() == type2attr<ElementC>, "input a wrong data type for matrix D \n");

    auto m  = matrix_A.size(0);
    auto k  = matrix_A.size(1);
    auto k2 = matrix_B.size(0);
    auto n  = matrix_B.size(1);
    
    auto mC = matrix_C.size(0);
    auto nC = matrix_C.size(1);

    auto mD = matrix_D.size(0);
    auto nD = matrix_D.size(1);

    // check the hidden dimension - k
    if (k != k2)
    {
        std::stringstream s; 
        s << "cannot apply matrix multiplication between two shapes: A=(" << m << ", " << k << ") and B=(" << k2 << ", " << n << ") \n";
        TORCH_CHECK(false, s.str());
    }

    // check shape match - A * B, C
    if (m != mC || n != nC)
    {
        std::stringstream s;
        s << "matrix A * B cannot add with matrix C between two shapes: A*B=(" << m << ", " << n << ") and C=(" << mC << ", " << nC << ") \n";
        TORCH_CHECK(false, s.str());
    }

    // check shape match - A * B, D
    if (m != mD || n != nD)
    {
        std::stringstream s;
        s << "matrix A * B cannot add with matrix D between two shapes: A*B=(" << m << ", " << n << ") and D=(" << mD << ", " << nD << ") \n";
        TORCH_CHECK(false, s.str());
    }

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{
                                    {m, n, k}, 
                                    {reinterpret_cast<ElementA*>(matrix_A.data()), k},  
                                    {reinterpret_cast<ElementB*>(matrix_B.data()), n},
                                    {reinterpret_cast<ElementC*>(matrix_C.data()), n},
                                    {reinterpret_cast<ElementC*>(matrix_D.data()), n},
                                    {alpha, beta},
                                    split_k_slices
                                };     

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op(arguments, workspace.get());
    CUTLASS_CHECK(status);

    cudaDeviceSynchronize();
}

// // // // // // // // // // // // // // // // // // // // // // // // 


#define HANDLE_OTHER_TYPES { \
    std::stringstream s; \
    s << "not implemented for (" \
      << "arch = " << arch \
      << ", " \
      << "dtype = " << type2str<CutlassType> \
      << ") " << std::endl; \
    TORCH_CHECK(false, s.str()); }


// #define COMPILE_CC (80)
inline bool is_available(int arch, int target)
{
    if (arch >= target)
    {
        if (target == 80 && COMPILE_CC >= 80) return true;
        if (target == 75 && COMPILE_CC >= 75) return true;
    }
    return false;
}

void GEMM(
        const torch::Tensor& matrix_A,
        const torch::Tensor& matrix_B,
        const torch::Tensor& matrix_C,
        const torch::Tensor& matrix_D,
        float alpha = 1.0, 
        float beta = 0.0
    ){
    // NOTE: in/out data types must be the same
    auto torch_type = matrix_A.scalar_type();
    auto props = getDeviceProps();

    // check cuda / arch
    int arch = props.major * 10 + props.minor;
    
    // dispatch: fp16, fp32, fp64
    if (torch_type == at::ScalarType::Half) {
        using CutlassType = cutlass::half_t;
        if (is_available(arch, 80)) {
            using Gemm = KernelConfig<CutlassType, 80>::Gemm;
            GEMM_kernel<Gemm>(matrix_A, matrix_B, matrix_C, matrix_D, alpha, beta);
        } else if (is_available(arch, 75)) {
            using Gemm = KernelConfig<CutlassType, 75>::Gemm;
            GEMM_kernel<Gemm>(matrix_A, matrix_B, matrix_C, matrix_D, alpha, beta);
        }
        else HANDLE_OTHER_TYPES
    } else if (torch_type == at::ScalarType::Float) {
        using CutlassType = float;
        if (is_available(arch, 80)) {
            using Gemm = KernelConfig<CutlassType, 80>::Gemm;
            GEMM_kernel<Gemm>(matrix_A, matrix_B, matrix_C, matrix_D, alpha, beta);
        } else if (is_available(arch, 50)) {
            using Gemm = KernelConfig<CutlassType, 50>::Gemm;
            GEMM_kernel<Gemm>(matrix_A, matrix_B, matrix_C, matrix_D, alpha, beta);
        }
        else HANDLE_OTHER_TYPES
    } else {
        TORCH_CHECK(false, "not implemented for this data type \n");
    }
}


// // // // // // // // // // // // // // // // // // // // // // // //
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // pytorch uses row-major
    m.def("GEMM", &GEMM,
          "GEMM (CUDA)", 
          py::arg("matrix_A"), 
          py::arg("matrix_B"), 
          py::arg("matrix_C"), 
          py::arg("matrix_D"),
          py::arg("alpha"),
          py::arg("beta")
        );
}
