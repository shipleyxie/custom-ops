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

#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
public:
    explicit ZeroOutOp(OpKernelConstruction* context)
        : OpKernel(context)
    {
    }

    void Compute(OpKernelContext* context) override
    {

        // make sure we have 2 inputs
        // 1-d input and lfilter coeffs a b.
        DCHECK_EQ(2, context->num_inputs());

        // get input tensor
        const Tensor& input = context->input(0);
        // get the weight tensor
        const Tensor& weights = context->input(1);

        // check the shape of inputs and weights
        const TensorShape& input_shape = input.shape();
        const TensorShape& weights_shape = weights.shape();
        DCHECK_EQ(input_shape.dims(), 2);
        // DCHECK_EQ(input_shape.dim_size(1), 1);
        DCHECK_EQ(weights_shape.dims(), 2);
        DCHECK_EQ(weights_shape.dim_size(0), 2);
        // DCHECK_EQ(weights_shape.dim_size(1), 1);

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(input_shape.dim_size(1));

        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        // get the corresponding Eigen tensors for data access
        auto input_tensor = input.matrix<float>();
        auto weights_tensor = weights.matrix<float>();
        auto output_tensor = output->matrix<float>();

        for (int i = 0; i < input_shape.dim_size(0); i++) { // iterate batch
            float temp = 0;
            output_tensor(i, 0) = weights_tensor(0, 0) * input_tensor(i, 0)
                + weights_tensor(1, 0) * temp;
            temp = output_tensor(i, 0);

            for (int j = 1; j < input_shape.dim_size(1); j++) {
                output_tensor(i, j) = weights_tensor(0, 0) * input_tensor(i, j)
                    + weights_tensor(1, 0) * temp;
                temp = output_tensor(i, j);
            }

            // calc PCEN
            float alpha = 1.0;
            float delta = 2;
            float r = 0.5;
            float s = 0.025;
            double eps = 1e-20f;
            double temp1;
            for (int j = 0; j < input_shape.dim_size(1); j++) {
                temp1 = (double)pow((double)(eps + output_tensor(i, j)), -alpha)
                        * input_tensor(i, j)
                    + delta;
                output_tensor(j, 0)
                    = (float)(pow((double)temp1, r) - pow((double)delta, r));
            }
        }
        /*
         * Usage:
         * python3 -c "import tensorflow as tf; tf.enable_eager_execution();import
         tensorflow_zero_out;
         print(tensorflow_zero_out.zero_out([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],[[0.025],[0.975]]))"
         tf.Tensor(
         [[0.025     ]
         [0.074375  ]
         [0.14751562]
         [0.24382773]
         [0.36273205]
         [0.5036638 ]
         [0.6660722 ]
         [0.84942037]
         [1.0531849 ]
         [1.2768552 ]], shape=(10, 1), dtype=float32)
         */
    }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
