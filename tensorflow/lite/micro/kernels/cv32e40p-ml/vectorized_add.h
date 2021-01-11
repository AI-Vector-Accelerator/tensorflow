#ifndef VECTORIZED_ADD_H_
#define VECTORIZED_ADD_H_

#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "fixedpoint/fixedpoint.h"

#include <vector>
#include "vector_operations.h"

namespace tflite {

inline void CheckArithmeticParams(const ArithmeticParams& params) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  // Input offset is negative input zero point. Activation tensors are
  // asymmetric quantized so they span the full int8 range.
  TFLITE_DCHECK_GE(-params.input1_offset, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_GE(-params.input2_offset, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_LE(-params.input1_offset, std::numeric_limits<int8_t>::max());
  TFLITE_DCHECK_LE(-params.input2_offset, std::numeric_limits<int8_t>::max());
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const int8_t* input1_data, const int8_t* input2_data,
                           int8_t* output_data) {
  CheckArithmeticParams(params);
  std::vector<int32_t> input1_temp, input2_temp;
  input1_temp.resize(size);
  input2_temp.resize(size);

  vect_addElementWise(size, input1_data, &input1_temp[0], params.input1_offset, (1 << params.left_shift));
  vect_addElementWise(size, input2_data, &input2_temp[0], params.input2_offset, (1 << params.left_shift));
  for (int i = 0; i < size; ++i) {
    input1_temp[i] = MultiplyByQuantizedMultiplierSmallerThanOneExp(input1_temp[i], params.input1_multiplier, params.input1_shift);
    input2_temp[i] = MultiplyByQuantizedMultiplierSmallerThanOneExp(input2_temp[i], params.input2_multiplier, params.input2_shift);
  }  
  vect_add_32bits(size, &input1_temp[0], &input2_temp[0], &input1_temp[0]);
  for (int i = 0; i < size; ++i) {  
    input1_temp[i] = MultiplyByQuantizedMultiplierSmallerThanOneExp(input1_temp[i], params.output_multiplier, params.output_shift) + params.output_offset;
  }
  vect_ReLu6_Bound_32bits(size,  &input1_temp[0] , &input1_temp[0], params.quantized_activation_min, params.quantized_activation_max);
  for (int i = 0; i < size; ++i) {  
    output_data[i] = static_cast<int8_t>(input1_temp[i]);
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  CheckArithmeticParams(params);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}










}


#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING_H_
