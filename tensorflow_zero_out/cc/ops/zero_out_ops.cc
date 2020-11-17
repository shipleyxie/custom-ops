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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: float")
    .Input("weights: float")
    .Output("zeroed: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

      shape_inference::ShapeHandle weight_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));

      shape_inference::DimensionHandle output_rows = c->Dim(input_shape, 0);
      shape_inference::DimensionHandle output_col = c->Dim(input_shape, 1);

      //shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
      //shape_inference::DimensionHandle weights_col = c->Dim(weight_shape, 1);
      //shape_inference::DimensionHandle merged;
      //TF_RETURN_IF_ERROR(c->Merge(input_rows, weights_col, &merged));


      c->set_output(0, c->Matrix(output_rows, output_col));
      return Status::OK();
    });
