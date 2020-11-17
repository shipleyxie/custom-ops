/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <iostream>
#include <ostream>
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "time_two.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T> struct ZeroOutFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &d, int batch_size, int data_len,
                  const T *input_tensor, const T *weights_tensor,
                  T *output_tensor) {

      VLOG(3) << "batch_size is "<< batch_size
          << " data_len is " << data_len;
    VLOG(3) << "weights_tensor[0,0]" << weights_tensor[0,0] 
        << "weights_tensor[0,1]"  << weights_tensor[0,1];

    for (int i = 0; i < batch_size; i++) { // iterate batch
      float temp = 0;
      output_tensor[i, 0] = weights_tensor[0, 0] * input_tensor[i, 0] +
                            weights_tensor[0, 1] * temp;
      VLOG(3) << "output_tensor[" << i << ", 0] = " << weights_tensor[0,0]
          << "*" << input_tensor[i,0] << "+" << weights_tensor[0,1] << "*"
          << temp << ";";
      VLOG(3) << "output_tensor result is " << output_tensor[i,0];

      temp = output_tensor[i, 0];

      for (int j = 1; j < data_len; j++) {
        output_tensor[i, j] = weights_tensor[0, 0] * input_tensor[i, j] +
                              weights_tensor[0, 1] * temp;


      VLOG(3) << "output_tensor[" << i << "," << j  <<  " ] = " << weights_tensor[0,0]
          << "*" << input_tensor[i,j] << "+" << weights_tensor[0,1] << "*"
          << temp << ";";
      VLOG(3) << "output_tensor result is " << output_tensor[i,j];
        temp = output_tensor[i, j];
      }

      // calc PCEN
      //float alpha = 1.0;
      //float delta = 2;
      //float r = 0.5;
      //float s = 0.025;
      //double eps = 1e-20f;
      //double temp1;
      //for (int j = 0; j < data_len; j++) {
        //temp1 = (double)pow((double)(eps + output_tensor[i, j]), -alpha) *
                    //input_tensor[i, j] +
                //delta;
        //output_tensor[j, 0] =
            //(float)(pow((double)temp1, r) - pow((double)delta, r));
      //}
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T> class ZeroOutOp : public OpKernel {
public:
  explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Grab the input tensor
    DCHECK_EQ(2, context->num_inputs());
    const Tensor &input_tensor = context->input(0);
    const Tensor &weights_tensor = context->input(1);

    const TensorShape &input_shape = input_tensor.shape();
    const TensorShape &weights_shape = weights_tensor.shape();

    DCHECK_EQ(input_shape.dims(), 2);
    LOG(INFO) << input_shape.dims();
    DCHECK_EQ(weights_shape.dim_size(0), 2);

    // Create an output tensor
    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ZeroOutFunctor<Device, T>()(
        context->eigen_device<Device>(), input_shape.dim_size(0),
        input_shape.dim_size(1), input_tensor.matrix<T>().data(),
        weights_tensor.matrix<T>().data(), output_tensor->matrix<T>().data());
    VLOG(3) << "debug out_tensor <<<<< "
        << output_tensor->matrix<T>().data()[0,0]
        << output_tensor->matrix<T>().data()[0,1]
        << output_tensor->matrix<T>().data()[0,2]
        << output_tensor->matrix<T>().data()[0,3]
        << output_tensor->matrix<T>().data()[1,0]
        << output_tensor->matrix<T>().data()[1,1]
        << output_tensor->matrix<T>().data()[1,3]
        << output_tensor->matrix<T>().data()[2,0]
        << output_tensor->matrix<T>().data()[2,1]
        << output_tensor->matrix<T>().data()[2,2]
        << output_tensor->matrix<T>().data()[2,3];

  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<T>("T"),               \
      ZeroOutOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                        \
  extern template struct ZeroOutFunctor<GPUDevice, T>;                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("ZeroOut").Device(DEVICE_GPU).TypeConstraint<T>("T"),               \
      ZeroOutOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif // GOOGLE_CUDA
} // namespace functor
} // namespace tensorflow
