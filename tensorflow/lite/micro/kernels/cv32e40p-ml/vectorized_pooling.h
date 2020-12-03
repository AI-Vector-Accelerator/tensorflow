#ifndef VECTORIZED_POOLING_H_
#define VECTORIZED_POOLING_H_

#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

#include "vector_operations.h"

namespace tflite {


inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const uint8_t* input_data,
                        const RuntimeShape& output_shape,
                        uint8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min, params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int32_t vecN;
  const uint32_t vect_data_stride=depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin = (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin = (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end = std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end = std::min(params.filter_height, input_height - in_y_origin);
          vecN=filter_x_end-filter_x_start;
          int32_t acc = 0;
          uint16_t tempAcc=0;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
            const int in_x = in_x_origin; 
            const int in_y = in_y_origin + filter_y;
            const uint8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, channel)];
            vectu_addReduction_stride(vecN, input_val, &tempAcc, tempAcc, vect_data_stride);
            filter_count+=vecN;
          }
          acc=static_cast<int32_t>(tempAcc);
          acc = (acc + filter_count / 2) / filter_count;
          acc = std::max(acc, params.quantized_activation_min);
          acc = std::min(acc, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] = static_cast<uint8_t>(acc);
        }
      }
    }
  }
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const int8_t* input_data,
                        const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min, params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int32_t vecN;
  const uint32_t vect_data_stride=depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin = (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin = (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end = std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end = std::min(params.filter_height, input_height - in_y_origin);
          vecN=filter_x_end-filter_x_start;
          int32_t acc = 0;
          int16_t tempAcc=0;
          int filter_count = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
            const int in_x = in_x_origin; 
            const int in_y = in_y_origin + filter_y;
            const int8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, channel)];
            vect_addReduction_stride(vecN, input_val, &tempAcc, tempAcc, vect_data_stride);
            filter_count+=vecN;
          }
          acc=static_cast<int32_t>(tempAcc);
          // Round to the closest integer value.
          acc = acc > 0 ? (acc + filter_count / 2) / filter_count : (acc - filter_count / 2) / filter_count;
          acc = std::max(acc, params.quantized_activation_min);
          acc = std::min(acc, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] = static_cast<int8_t>(acc);
        }
      }
    }
  }
}


inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const uint8_t* input_data, const RuntimeShape& output_shape,
                    uint8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min, params.quantized_activation_max);
  TFLITE_DCHECK_GE(params.quantized_activation_min, 0);
  TFLITE_DCHECK_LE(params.quantized_activation_max, 255);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int32_t vecN;
  const uint32_t vect_data_stride=depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin = (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin = (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end = std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end = std::min(params.filter_height, input_height - in_y_origin);
          vecN=filter_x_end-filter_x_start;
          uint8_t max = 0;
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
            const int in_x = in_x_origin; 
            const int in_y = in_y_origin + filter_y;
            const uint8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, channel)];
            vectu_maxReduction_stride(vecN, input_val, &max, max, vect_data_stride);
          }
          max = std::max<uint8_t>(max, params.quantized_activation_min);
          max = std::min<uint8_t>(max, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] = static_cast<uint8_t>(max);
        }
      }
    }
  }
}



inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const int8_t* input_data, const RuntimeShape& output_shape,
                    int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min, params.quantized_activation_max);
  TFLITE_DCHECK_GE(params.quantized_activation_min, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_LE(params.quantized_activation_max, std::numeric_limits<int8_t>::max());
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  int32_t vecN;
  const uint32_t vect_data_stride=depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const int in_x_origin = (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin = (out_y * stride_height) - params.padding_values.height;
          // Compute the boundaries of the filter region clamped so as to
          // ensure that the filter window fits in the input array.
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end = std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end = std::min(params.filter_height, input_height - in_y_origin);
          vecN=filter_x_end-filter_x_start;
          int8_t max = std::numeric_limits<int8_t>::lowest();
          for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
            const int in_x = in_x_origin; 
            const int in_y = in_y_origin + filter_y;
            const int8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, channel)];
            vect_maxReduction_stride(vecN, input_val, &max, max, vect_data_stride);
          }
          max = std::max<int8_t>(max, params.quantized_activation_min);
          max = std::min<int8_t>(max, params.quantized_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, channel)] = static_cast<int8_t>(max);
        }
      }
    }
  }
}









}


#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_POOLING_H_
