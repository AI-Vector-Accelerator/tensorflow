#ifndef VECTORIZED_CONV_H_
#define VECTORIZED_CONV_H_

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "vector_operations.h"

namespace tflite {


inline void ConvPointwise(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8_t* input_data, const RuntimeShape& filter_shape,
                 const uint8_t* filter_data, const RuntimeShape& bias_shape,
                 const int32_t* bias_data, const RuntimeShape& output_shape,
                 uint8_t* output_data, const RuntimeShape& im2col_shape,
                 uint8_t* im2col_data, void* cpu_backend_context) {
  (void)cpu_backend_context;  // only used in optimized code.
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  int32_t vecN=input_depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          int32_t acc = 0, tempAcc;
          const int in_y = in_y_origin;
          const int in_x = in_x_origin;
          int filter_x = 0;
          int filter_y = 0;
          int in_channel = 0;

          const uint8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
          const uint8_t* filter_val = &filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
          vectu_dotProduct_offset(vecN, input_val, filter_val, &tempAcc, input_offset, filter_offset);
				  acc +=tempAcc; 
          
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<uint8_t>(acc);
        }
      }
    }
  }
}


 void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8_t* input_data, const RuntimeShape& filter_shape,
                 const uint8_t* filter_data, const RuntimeShape& bias_shape,
                 const int32_t* bias_data, const RuntimeShape& output_shape,
                 uint8_t* output_data, const RuntimeShape& im2col_shape,
                 uint8_t* im2col_data, void* cpu_backend_context) {
  if(filter_shape.Dims(1)==1 && filter_shape.Dims(2)==1){//pointwise operation, use edge case pointwise function for optimal use of vector extension hardware
    ConvPointwise(params, input_shape, input_data,  filter_shape,filter_data, bias_shape, bias_data,  output_shape, output_data, im2col_shape, im2col_data,  cpu_backend_context);
    return;
  }   
  (void)cpu_backend_context;  // only used in optimized code.
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor; 
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;  
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const uint32_t vect_data_stride=dilation_width_factor*input_depth;
  const uint32_t vect_filter_stride=input_depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
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
		
			      for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
				      const uint8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
              const uint8_t* filter_val = &filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
              vectu_dotProduct_offset_stride(vecN, input_val, filter_val, &tempAcc, input_offset, filter_offset, vect_data_stride,vect_filter_stride);
				      acc +=tempAcc;
            }
          }

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<uint8_t>(acc);
        }
      }
	  }
  }
}


inline void ConvPerChannelPointwise(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
   // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const int32_t filter_offset = 0;
  int32_t vecN=input_depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          int32_t acc = 0, tempAcc;
          const int in_y = in_y_origin;
          const int in_x = in_x_origin;
          int filter_x = 0;
          int filter_y = 0;
          int in_channel = 0;

          const int8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
          const int8_t* filter_val = &filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
          vect_dotProduct_offset(vecN, input_val, filter_val, &tempAcc, input_offset, filter_offset);
				  acc +=tempAcc; 

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<int8_t>(acc);
        }
      }
    }
  }
}


inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  
  if(filter_shape.Dims(1)==1 && filter_shape.Dims(2)==1){//pointwise operation, use edge case pointwise function for optimal use of vector extension hardware
    ConvPerChannelPointwise(params, output_multiplier, output_shift, input_shape,input_data,  filter_shape,filter_data,  bias_shape,bias_data,  output_shape, output_data);
    return;
  }    
  // Get parameters.
  const int32_t input_offset = (params.input_offset);  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const uint32_t vect_data_stride=dilation_width_factor*input_depth;
  const uint32_t vect_filter_stride=input_depth;
  const int32_t filter_offset = 0;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
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

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                // Accumulate with 32 bits accumulator.
                // In the nudging process during model quantization, we force
                // real value of 0.0 be represented by a quantized value. This
                // guarantees that the input_offset is a int8_t, even though
                // it is represented using int32_t. int32_t += int8_t *
                // (int8_t - int8_t) so the highest value we can get from each
                // accumulation is [-127, 127] * ([-128, 127] -
                // [-128, 127]), which is [-32512, 32512]. log2(32512)
                // = 14.98, which means we can accumulate at least 2^16
                // multiplications without overflow. The accumulator is
                // applied to a filter so the accumulation logic will hold as
                // long as the filter size (filter_y * filter_x * in_channel)
                // does not exceed 2^16, which is the case in all the models
                // we have seen so far.
                // TODO(jianlijianli): Add a check to make sure the
                // accumulator depth is smaller than 2^16.
              const int8_t* input_val = &input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
              const int8_t* filter_val = &filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
              vect_dotProduct_offset_stride(vecN, input_val, filter_val, &tempAcc, input_offset, filter_offset, vect_data_stride,vect_filter_stride);
				      acc +=tempAcc;
            }
          }

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<int8_t>(acc);
        }
      }
    }
  }
}







}

#endif  
