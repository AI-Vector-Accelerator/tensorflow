#ifndef VECTORIZED_DEPTHWISE_CONV_H_
#define VECTORIZED_DEPTHWISE_CONV_H_

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/common.h"

#include "vector_operations.h"


namespace tflite {


namespace depthwise_conv {

template <DepthwiseConvOutputRounding output_rounding>
inline int32_t DepthwiseConvRound(int32_t x, int32_t quantized_multiplier,
                                  int shift) {
  TFLITE_DCHECK_NE(output_rounding, DepthwiseConvOutputRounding::kNone);
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

template <>
inline int32_t DepthwiseConvRound<DepthwiseConvOutputRounding::kAwayFromZero>(
    int32_t x, int32_t quantized_multiplier, int shift) {
  return MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift);
}

template <>
inline int32_t DepthwiseConvRound<DepthwiseConvOutputRounding::kUpward>(
    int32_t x, int32_t quantized_multiplier, int shift) {
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  const int rounding_offset = right_shift > 0 ? 1 << (right_shift - 1) : 0;
  return (SaturatingRoundingDoublingHighMul(x * (1 << left_shift),
                                            quantized_multiplier) +
          rounding_offset) >>
         right_shift;
}


template <DepthwiseConvOutputRounding output_rounding>
struct DepthwiseConvBasicKernel {
  static inline void Run(
      const DepthwiseParams& params, const RuntimeShape& input_shape,
      const uint8_t* input_data, const RuntimeShape& filter_shape,
      const uint8_t* filter_data, const RuntimeShape& bias_shape,
      const int32_t* bias_data, const RuntimeShape& output_shape,
      uint8_t* output_data) {
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    const int32_t input_offset = params.input_offset;
    const int32_t filter_offset = params.weights_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_multiplier = params.output_multiplier;
    const int output_shift = params.output_shift;
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    const uint32_t vect_data_stride=dilation_width_factor*input_depth;
    const uint32_t vect_filter_stride=output_depth;

    for (int b = 0; b < batches; ++b) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int ic = 0; ic < input_depth; ++ic) {
            for (int m = 0; m < depth_multiplier; m++) {
              const int oc = m + ic * depth_multiplier;
              const int in_x_origin = (out_x * stride_width) - pad_width;
              const int in_y_origin = (out_y * stride_height) - pad_height;
              int32_t acc = 0, tempAcc;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                const int in_y = in_y_origin + dilation_height_factor * filter_y;
                int filter_x=0;  
                int in_x = in_x_origin;

                // Zero padding by omitting the rows outside the image.
                const bool is_point_inside_image = (in_y >= 0) && (in_y < input_height);
                if (!is_point_inside_image){continue;}
			
			          // Zero padding by reducing filter size if filter extends outside the image.
			          int32_t vecN=filter_width;
			          if(in_x<0){
				          vecN+=in_x;
				          filter_x=-in_x;
				          in_x=0;
			          }else if(in_x+filter_width>input_width){
				          vecN=input_width-in_x;
			          }

                const uint8_t* input_val = &input_data[Offset(input_shape, b, in_y, in_x, ic)];
                const uint8_t* filter_val = &filter_data[Offset( filter_shape, 0, filter_y, filter_x, oc)];
                vectu_dotProduct_offset_stride(vecN, input_val, filter_val, &tempAcc, input_offset, filter_offset, vect_data_stride,vect_filter_stride);
				        acc +=tempAcc;
              }
              
              if (bias_data) {
                acc += bias_data[oc];
              }
              acc = DepthwiseConvRound<output_rounding>(acc, output_multiplier, output_shift);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[Offset(output_shape, b, out_y, out_x, oc)] = static_cast<uint8_t>(acc);
            }
          }
        }
      }
    }
  }

  // TODO(b/148596273): Reconcile reference versions, perhaps with common
  // MultiplyByQuantizedMultiplier or DepthwiseConvRound function.
  static inline void RunPerChannel(
      const DepthwiseParams& params, const RuntimeShape& input_shape,
      const int8_t* input_data, const RuntimeShape& filter_shape,
      const int8_t* filter_data, const RuntimeShape& bias_shape,
      const int32_t* bias_data, const RuntimeShape& output_shape,
      int8_t* output_data) {
    // Get parameters.
    // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t input_offset = params.input_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;
    const int32_t* output_multiplier = params.output_multiplier_per_channel;
    const int32_t* output_shift = params.output_shift_per_channel;

    // Check dimensions of the tensors.
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

    const uint32_t vect_data_stride=dilation_width_factor*input_depth;
    const uint32_t vect_filter_stride=output_depth;
    const int32_t filter_offset = 0;

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
            for (int m = 0; m < depth_multiplier; ++m) {
              const int output_channel = m + in_channel * depth_multiplier;
              const int in_x_origin = (out_x * stride_width) - pad_width;
              const int in_y_origin = (out_y * stride_height) - pad_height;
              int32_t acc = 0, tempAcc;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                const int in_y = in_y_origin + dilation_height_factor * filter_y;
                int filter_x=0;  
                int in_x = in_x_origin; 
                 
                 
                 // Zero padding by omitting the rows outside the image.
                const bool is_point_inside_image = (in_y >= 0) && (in_y < input_height);
                if (!is_point_inside_image){continue;}
			
			          // Zero padding by reducing filter size if filter extends outside the image.
			          int32_t vecN=filter_width;
			          if(in_x<0){
				          vecN+=in_x;
				          filter_x=-in_x;
				          in_x=0;
			          }else if(in_x+filter_width>input_width){
				          vecN=input_width-in_x;
			          }

                const int8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
                const int8_t* filter_val = &filter_data[Offset( filter_shape, 0, filter_y, filter_x, output_channel)];
                vect_dotProduct_offset_stride(vecN, input_val, filter_val, &tempAcc, input_offset, filter_offset, vect_data_stride,vect_filter_stride);
				        acc +=tempAcc; 
              }

              if (bias_data) {
                acc += bias_data[output_channel];
              }
              acc = DepthwiseConvRound<output_rounding>(acc, output_multiplier[output_channel],output_shift[output_channel]);
              acc += output_offset;
              acc = std::max(acc, output_activation_min);
              acc = std::min(acc, output_activation_max);
              output_data[Offset(output_shape, batch, out_y, out_x, output_channel)] = static_cast<int8_t>(acc);
            }
          }
        }
      }
    }
  }
};

}

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8_t* input_data, const RuntimeShape& filter_shape,
    const uint8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    uint8_t* output_data) {
  return depthwise_conv::DepthwiseConvBasicKernel<
      DepthwiseConvOutputRounding::kAwayFromZero>::Run(params, input_shape,
                                                       input_data, filter_shape,
                                                       filter_data, bias_shape,
                                                       bias_data, output_shape,
                                                       output_data);
}




inline void DepthwiseConvPerChannel(
    const DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

  const uint32_t vect_data_stride=dilation_width_factor*input_depth;
  const uint32_t vect_filter_stride=output_depth;
  const int32_t filter_offset = 0;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            const int output_channel = m + in_channel * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32_t acc = 0, tempAcc;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              int filter_x=0;  
              int in_x = in_x_origin; 
                 
                 
              // Zero padding by omitting the rows outside the image.
              const bool is_point_inside_image = (in_y >= 0) && (in_y < input_height);
              if (!is_point_inside_image){continue;}
			
			        // Zero padding by reducing filter size if filter extends outside the image.
			        int32_t vecN=filter_width;
			        if(in_x<0){
				        vecN+=in_x;
				        filter_x=-in_x;
				        in_x=0;
			        }else if(in_x+filter_width>input_width){
				        vecN=input_width-in_x;
			        }

              const int8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
              const int8_t* filter_val = &filter_data[Offset( filter_shape, 0, filter_y, filter_x, output_channel)];
              vect_dotProduct_offset_stride(vecN, input_val, filter_val, &tempAcc, input_offset, filter_offset, vect_data_stride,vect_filter_stride);
				      acc +=tempAcc; 
            }

            if (bias_data) {
              acc += bias_data[output_channel];
            }
            acc = MultiplyByQuantizedMultiplier( acc, output_multiplier[output_channel], output_shift[output_channel]);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, batch, out_y, out_x, output_channel)] = static_cast<int8_t>(acc);
          }
        }
      }
    }
  }
}



}

#endif  
