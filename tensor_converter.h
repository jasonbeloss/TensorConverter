/*
 * MIT License
 *
 * Copyright (c) 2024 TensorConverter
 * Author: Jiacheng.Du
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef TENSOR_CONVERTER_H
#define TENSOR_CONVERTER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

// Error message macro definitions
#define ERROR_MSG_SIZE 256
#define ERROR_MSG_SUCCESS "Conversion successful"
#define ERROR_MSG_NULL_POINTER "Input pointer is null"
#define ERROR_MSG_INVALID_DIMS "Invalid dimension parameters"
#define ERROR_MSG_UNSUPPORTED_TYPE "Unsupported data type"
#define ERROR_MSG_MEMORY_ALLOC "Memory allocation failed"
#define ERROR_MSG_LAYOUT_CONVERSION "Layout conversion failed"
#define ERROR_MSG_DATA_COPY "Data copy failed"
#define ERROR_MSG_INVALID_LAYOUT "Invalid layout format"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Data type enumeration
 */
typedef enum {
    TENSOR_FLOAT32 = 0,
    TENSOR_INT32 = 1,
    TENSOR_UINT8 = 2,
    TENSOR_INT64 = 3,
    TENSOR_INT16 = 6,
    TENSOR_INT8 = 8,
    TENSOR_FLOAT16 = 9
} tensor_data_type_t;

/**
 * Tensor layout enumeration
 */
typedef enum {
    LAYOUT_UNKNOWN = 0,
    LAYOUT_NCHW = 1,    // ONNX common format (Batch, Channel, Height, Width)
    LAYOUT_NHWC = 2,    // TFLite common format (Batch, Height, Width, Channel)
    LAYOUT_GENERIC = 3  // Other dimension layouts, no conversion needed
} tensor_layout_t;

/**
 * Tensor dimension information structure
 */
typedef struct {
    int32_t* dims;           // Dimension array
    size_t num_dims;         // Number of dimensions
    tensor_data_type_t data_type; // Data type
    size_t total_elements;   // Total number of elements
    tensor_layout_t layout;     // Tensor layout format
} tensor_shape_t;

/**
 * Conversion result structure
 */
typedef struct {
    void* data;              // Pointer to converted data
    size_t data_size;        // Data size (bytes)
    tensor_shape_t shape;       // Tensor shape information
    bool success;            // Whether conversion was successful
    char error_msg[ERROR_MSG_SIZE];     // Error message
} conversion_result_t;

/**
 * Get byte size of data type
 * @param data_type Data type
 * @return Byte size, returns 0 if type is not supported
 */
static inline size_t get_data_type_size(tensor_data_type_t data_type);

/**
 * Calculate total number of elements in tensor
 * @param dims Dimension array
 * @param num_dims Number of dimensions
 * @return Total number of elements
 */
static inline size_t calculate_total_elements(const int32_t* dims, size_t num_dims);

/**
 * Validate tensor shape validity
 * @param dims Dimension array
 * @param num_dims Number of dimensions
 * @return Returns true if valid, false otherwise
 */
static inline bool validate_tensor_shape(const int32_t* dims, size_t num_dims);



/**
 * Free memory allocated in conversion result
 * @param result Conversion result pointer
 */
static inline void free_conversion_result(conversion_result_t* result);

/**
 * Copy tensor data (generic function)
 * @param src Source data pointer
 * @param dst Destination data pointer
 * @param element_size Single element size
 * @param total_elements Total number of elements
 * @return Returns true if successful, false otherwise
 */
static inline bool copy_tensor_data(const void* src, void* dst, size_t element_size, size_t total_elements);

/**
 * Print tensor information (for debugging)
 * @param shape Tensor shape information
 */
static inline void print_tensor_info(const tensor_shape_t* shape);

/**
 * Safe error message formatting function
 * @param buffer Error message buffer
 * @param buffer_size Buffer size
 * @param format Format string
 * @param ... Variable arguments
 * @return Number of characters written (excluding null terminator)
 */
static inline int safe_snprintf(char* buffer, size_t buffer_size, const char* format, ...) {
    if (!buffer || buffer_size == 0 || !format) {
        return -1;
    }
    va_list args;
    va_start(args, format);
    int result = vsnprintf(buffer, buffer_size, format, args);
    va_end(args);
    // Ensure null termination
    if (result >= 0 && (size_t)result >= buffer_size) {
        buffer[buffer_size - 1] = '\0';
        result = (int)(buffer_size - 1);
    }
    return result;
}

/**
 * Validate data pointer validity
 * @param data Data pointer
 * @param expected_size Expected data size
 * @return Returns true if valid, false otherwise
 */
static inline bool validate_data_pointer(const void* data, size_t expected_size) {
    if (!data) {
        return false;
    }
    // Check if expected_size is reasonable (not too large)
    if (expected_size == 0 || expected_size > SIZE_MAX / 2) {
        return false;
    }
    // Note: We cannot actually validate if the pointer points to valid memory
    // without potentially causing a segmentation fault. This is a limitation
    // of C/C++. The caller should ensure the pointer is valid.
    return true;
}

/**
 * Validate memory boundaries for layout conversion
 * @param src Source data pointer
 * @param dst Destination data pointer
 * @param total_elements Total number of elements
 * @param element_size Size of each element
 * @return Returns true if boundaries are valid, false otherwise
 */
static inline bool validate_memory_boundaries(const void* src, const void* dst,
                                             size_t total_elements, size_t element_size) {
    if (!src || !dst) {
        return false;
    }
    // Check for overflow in total size calculation
    if (total_elements > SIZE_MAX / element_size) {
        return false;
    }
    size_t total_bytes = total_elements * element_size;
    (void)total_bytes; // Suppress unused variable warning
    // Allow self-copy but warn about potential issues
    // Self-copy is valid for layout conversion when src and dst are the same buffer
    // but the caller should ensure sufficient memory is allocated

    // Note: We cannot actually validate memory boundaries without potentially
    // causing a segmentation fault. The caller should ensure sufficient memory
    // is allocated for both source and destination.

    return true;
}

/**
 * Detect tensor layout type
 * @param dims Dimension array
 * @param num_dims Number of dimensions
 * @return Detected layout type
 */
static inline tensor_layout_t detect_tensor_layout(const int32_t* dims, size_t num_dims) {
    if (!dims || num_dims == 0) {
        return LAYOUT_UNKNOWN;
    }

    // 4D tensors may need layout conversion (NCHW vs NHWC)
    if (num_dims == 4) {
        // More sophisticated heuristic for layout detection
        // NCHW: [Batch, Channel, Height, Width] - typical for CNN
        // NHWC: [Batch, Height, Width, Channel] - typical for TFLite

        // Check if this looks like image data with typical characteristics
        int32_t batch_size = dims[0];
        int32_t dim1 = dims[1];
        int32_t dim2 = dims[2];
        int32_t dim3 = dims[3];
        (void)batch_size; // Suppress unused variable warning

        // Common image dimensions: height and width are usually larger than channels
        // and often powers of 2 or multiples of 8/16/32
        bool dim2_large = (dim2 >= 32 && (dim2 % 8 == 0 || dim2 % 16 == 0 || dim2 % 32 == 0));
        bool dim3_large = (dim3 >= 32 && (dim3 % 8 == 0 || dim3 % 16 == 0 || dim3 % 32 == 0));
        bool dim1_small = (dim1 <= 128); // Channel dimension is usually smaller

        // If dim1 is small and dim2/dim3 are large, likely NCHW
        if (dim1_small && dim2_large && dim3_large) {
            return LAYOUT_NCHW;
        }

        // If dim3 is small and dim1/dim2 are large, likely NHWC
        if (dim3 <= 128 && dim1 >= 32 && dim2 >= 32) {
            return LAYOUT_NHWC;
        }

        // If we can't determine with confidence, return UNKNOWN
        // This is safer than making a potentially wrong guess
        return LAYOUT_UNKNOWN;
    }

    // Other dimension tensors usually don't need layout conversion
    return LAYOUT_GENERIC;
}

/**
 * NCHW to NHWC layout conversion
 * @param src Source data pointer
 * @param dst Destination data pointer
 * @param N Batch size
 * @param C Number of channels
 * @param H Height
 * @param W Width
 * @param element_size Single element byte size
 * @return Whether conversion was successful
 */
static inline bool convert_nchw_to_nhwc(const void* src, void* dst,
                         int32_t N, int32_t C, int32_t H, int32_t W,
                         size_t element_size) {
    if (!src || !dst || N <= 0 || C <= 0 || H <= 0 || W <= 0 || element_size == 0) {
        return false;
    }

    // Check for potential overflow in index calculations
    if ((size_t)N > SIZE_MAX / (size_t)C / (size_t)H / (size_t)W ||
        (size_t)C > SIZE_MAX / (size_t)H / (size_t)W ||
        (size_t)H > SIZE_MAX / (size_t)W) {
        return false; // Overflow would occur
    }

    // Additional check: ensure total elements calculation doesn't overflow
    size_t total_elements = (size_t)N * C * H * W;
    if (total_elements == 0 || total_elements > SIZE_MAX / element_size) {
        return false;
    }

    // Validate memory boundaries
    if (!validate_memory_boundaries(src, dst, total_elements, element_size)) {
        return false;
    }

    const char* src_data = (const char*)src;
    char* dst_data = (char*)dst;

    // NCHW: [N][C][H][W] -> NHWC: [N][H][W][C]
    for (int32_t n = 0; n < N; n++) {
        for (int32_t h = 0; h < H; h++) {
            for (int32_t w = 0; w < W; w++) {
                for (int32_t c = 0; c < C; c++) {
                    // NCHW index: n*C*H*W + c*H*W + h*W + w
                    size_t src_idx = (size_t)n * C * H * W + (size_t)c * H * W + (size_t)h * W + (size_t)w;
                    // NHWC index: n*H*W*C + h*W*C + w*C + c
                    size_t dst_idx = (size_t)n * H * W * C + (size_t)h * W * C + (size_t)w * C + (size_t)c;

                    // Final safety check: ensure indices are within bounds
                    if (src_idx >= total_elements || dst_idx >= total_elements) {
                        return false;
                    }

                    memcpy(dst_data + dst_idx * element_size,
                           src_data + src_idx * element_size,
                           element_size);
                }
            }
        }
    }
    return true;
}

/**
 * NHWC to NCHW layout conversion
 * @param src Source data pointer
 * @param dst Destination data pointer
 * @param N Batch size
 * @param H Height
 * @param W Width
 * @param C Number of channels
 * @param element_size Single element byte size
 * @return Whether conversion was successful
 */
static inline bool convert_nhwc_to_nchw(const void* src, void* dst,
                         int32_t N, int32_t H, int32_t W, int32_t C,
                         size_t element_size) {
    if (!src || !dst || N <= 0 || H <= 0 || W <= 0 || C <= 0 || element_size == 0) {
        return false;
    }

    // Check for potential overflow in index calculations
    if ((size_t)N > SIZE_MAX / (size_t)H / (size_t)W / (size_t)C ||
        (size_t)H > SIZE_MAX / (size_t)W / (size_t)C ||
        (size_t)W > SIZE_MAX / (size_t)C) {
        return false; // Overflow would occur
    }
    // Additional check: ensure total elements calculation doesn't overflow
    size_t total_elements = (size_t)N * H * W * C;
    if (total_elements == 0 || total_elements > SIZE_MAX / element_size) {
        return false;
    }
    // Validate memory boundaries
    if (!validate_memory_boundaries(src, dst, total_elements, element_size)) {
        return false;
    }

    const char* src_data = (const char*)src;
    char* dst_data = (char*)dst;

    // NHWC: [N][H][W][C] -> NCHW: [N][C][H][W]
    for (int32_t n = 0; n < N; n++) {
        for (int32_t c = 0; c < C; c++) {
            for (int32_t h = 0; h < H; h++) {
                for (int32_t w = 0; w < W; w++) {
                    // NHWC index: n*H*W*C + h*W*C + w*C + c
                    size_t src_idx = (size_t)n * H * W * C + (size_t)h * W * C + (size_t)w * C + (size_t)c;
                    // NCHW index: n*C*H*W + c*H*W + h*W + w
                    size_t dst_idx = (size_t)n * C * H * W + (size_t)c * H * W + (size_t)h * W + (size_t)w;

                    // Final safety check: ensure indices are within bounds
                    if (src_idx >= total_elements || dst_idx >= total_elements) {
                        return false;
                    }

                    memcpy(dst_data + dst_idx * element_size,
                           src_data + src_idx * element_size,
                           element_size);
                }
            }
        }
    }
    return true;
}

/**
 * ONNX to TFLite conversion with layout conversion
 * @param onnx_data ONNX tensor data pointer
 * @param dims Tensor dimension array
 * @param num_dims Number of dimensions
 * @param data_type Data type
 * @param src_layout Source layout format
 * @param dst_layout Destination layout format
 * @return conversion_result_t Conversion result
 */
static inline conversion_result_t onnx_to_tflite_with_layout(const void* onnx_data,
                                                          const int32_t* dims,
                                                          size_t num_dims,
                                                          tensor_data_type_t data_type,
                                                          tensor_layout_t src_layout,
                                                          tensor_layout_t dst_layout) {
    conversion_result_t result = {0};

    // Validate input parameters
    if (!onnx_data || !dims || num_dims == 0) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_NULL_POINTER);
        return result;
    }

    if (!validate_tensor_shape(dims, num_dims)) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_INVALID_DIMS);
        return result;
    }

    size_t element_size = get_data_type_size(data_type);
    if (element_size == 0) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_UNSUPPORTED_TYPE ": %d", data_type);
        return result;
    }

    size_t total_elements = calculate_total_elements(dims, num_dims);
    if (total_elements == 0) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_INVALID_DIMS);
        return result;
    }

    // Check for overflow in total_bytes calculation
    if (total_elements > SIZE_MAX / element_size) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_INVALID_DIMS ": size too large");
        return result;
    }

    size_t total_bytes = element_size * total_elements;

    // Allocate memory for TFLite format data
    result.data = malloc(total_bytes);
    if (!result.data) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_MEMORY_ALLOC ": %zu bytes", total_bytes);
        return result;
    }

    // Allocate and copy dimension information
    result.shape.dims = (int32_t*)malloc(num_dims * sizeof(int32_t));
    if (!result.shape.dims) {
        free(result.data);
        result.data = NULL;
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_MEMORY_ALLOC);
        return result;
    }

    // Set result information
    result.shape.num_dims = num_dims;
    result.shape.data_type = data_type;
    result.shape.total_elements = total_elements;
    result.shape.layout = dst_layout;
    result.data_size = total_bytes;

    // Check if layout conversion is needed
    bool need_layout_conversion = false;
    if (num_dims == 4 && src_layout != dst_layout) {
        // Only conversion between NCHW and NHWC is supported
        if ((src_layout == LAYOUT_NCHW && dst_layout == LAYOUT_NHWC) ||
            (src_layout == LAYOUT_NHWC && dst_layout == LAYOUT_NCHW)) {
            need_layout_conversion = true;
        } else if (src_layout != LAYOUT_UNKNOWN && dst_layout != LAYOUT_UNKNOWN) {
            // Any other explicit layout conversion is not supported
            free(result.data);
            free(result.shape.dims);
            result.data = NULL;
            result.shape.dims = NULL;
            safe_snprintf(result.error_msg, sizeof(result.error_msg),
                    ERROR_MSG_LAYOUT_CONVERSION ": from %d to %d", src_layout, dst_layout);
            return result;
        }
    }

    if (need_layout_conversion) {
        // Layout conversion needed
        if (src_layout == LAYOUT_NCHW && dst_layout == LAYOUT_NHWC) {
            // NCHW -> NHWC
            memcpy(result.shape.dims, dims, num_dims * sizeof(int32_t));
            // Convert dimension order: [N,C,H,W] -> [N,H,W,C]
            // Keep N at index 0, move H to index 1, W to index 2, C to index 3
            result.shape.dims[1] = dims[2]; // H (was at index 2)
            result.shape.dims[2] = dims[3]; // W (was at index 3)
            result.shape.dims[3] = dims[1]; // C (was at index 1)

            if (!convert_nchw_to_nhwc(onnx_data, result.data,
                                     dims[0], dims[1], dims[2], dims[3],
                                     element_size)) {
                free(result.data);
                free(result.shape.dims);
                result.data = NULL;
                result.shape.dims = NULL;
                safe_snprintf(result.error_msg, sizeof(result.error_msg),
                        ERROR_MSG_LAYOUT_CONVERSION);
                return result;
            }
        } else if (src_layout == LAYOUT_NHWC && dst_layout == LAYOUT_NCHW) {
            // NHWC -> NCHW
            memcpy(result.shape.dims, dims, num_dims * sizeof(int32_t));
            // Convert dimension order: [N,H,W,C] -> [N,C,H,W]
            // Keep N at index 0, move C to index 1, H to index 2, W to index 3
            result.shape.dims[1] = dims[3]; // C (was at index 3)
            result.shape.dims[2] = dims[1]; // H (was at index 1)
            result.shape.dims[3] = dims[2]; // W (was at index 2)

            if (!convert_nhwc_to_nchw(onnx_data, result.data,
                                     dims[0], dims[1], dims[2], dims[3],
                                     element_size)) {
                free(result.data);
                free(result.shape.dims);
                result.data = NULL;
                result.shape.dims = NULL;
                safe_snprintf(result.error_msg, sizeof(result.error_msg),
                        ERROR_MSG_LAYOUT_CONVERSION);
                return result;
            }
        }
    } else {
        // No layout conversion needed, copy directly
        memcpy(result.shape.dims, dims, num_dims * sizeof(int32_t));
        if (!copy_tensor_data(onnx_data, result.data, element_size, total_elements)) {
            free(result.data);
            free(result.shape.dims);
            result.data = NULL;
            result.shape.dims = NULL;
            safe_snprintf(result.error_msg, sizeof(result.error_msg),
                    ERROR_MSG_DATA_COPY);
            return result;
        }
    }
    result.success = true;
    return result;
}

/**
 * TFLite to ONNX conversion with layout conversion
 * @param tflite_data TFLite tensor data pointer
 * @param dims Tensor dimension array
 * @param num_dims Number of dimensions
 * @param data_type Data type
 * @param src_layout Source layout format
 * @param dst_layout Destination layout format
 * @return conversion_result_t Conversion result
 */
static inline conversion_result_t tflite_to_onnx_with_layout(const void* tflite_data,
                                                          const int32_t* dims,
                                                          size_t num_dims,
                                                          tensor_data_type_t data_type,
                                                          tensor_layout_t src_layout,
                                                          tensor_layout_t dst_layout) {
    conversion_result_t result = {0};

    // Validate input parameters
    if (!tflite_data || !dims || num_dims == 0) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_NULL_POINTER);
        return result;
    }

    if (!validate_tensor_shape(dims, num_dims)) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_INVALID_DIMS);
        return result;
    }

    size_t element_size = get_data_type_size(data_type);
    if (element_size == 0) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_UNSUPPORTED_TYPE ": %d", data_type);
        return result;
    }

    size_t total_elements = calculate_total_elements(dims, num_dims);
    if (total_elements == 0) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_INVALID_DIMS);
        return result;
    }

    // Check for overflow in total_bytes calculation
    if (total_elements > SIZE_MAX / element_size) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_INVALID_DIMS ": size too large");
        return result;
    }

    size_t total_bytes = element_size * total_elements;

    // Allocate memory for ONNX format data
    result.data = malloc(total_bytes);
    if (!result.data) {
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_MEMORY_ALLOC ": %zu bytes", total_bytes);
        return result;
    }

    // Allocate and copy dimension information
    result.shape.dims = (int32_t*)malloc(num_dims * sizeof(int32_t));
    if (!result.shape.dims) {
        free(result.data);
        result.data = NULL;
        safe_snprintf(result.error_msg, sizeof(result.error_msg),
                ERROR_MSG_MEMORY_ALLOC);
        return result;
    }

    // Set result information
    result.shape.num_dims = num_dims;
    result.shape.data_type = data_type;
    result.shape.total_elements = total_elements;
    result.shape.layout = dst_layout;
    result.data_size = total_bytes;

    // Check if layout conversion is needed
    bool need_layout_conversion = false;
    if (num_dims == 4 && src_layout != dst_layout) {
        // Only conversion between NCHW and NHWC is supported
        if ((src_layout == LAYOUT_NCHW && dst_layout == LAYOUT_NHWC) ||
            (src_layout == LAYOUT_NHWC && dst_layout == LAYOUT_NCHW)) {
            need_layout_conversion = true;
        } else if (src_layout != LAYOUT_UNKNOWN && dst_layout != LAYOUT_UNKNOWN) {
            // Any other explicit layout conversion is not supported
            free(result.data);
            free(result.shape.dims);
            result.data = NULL;
            result.shape.dims = NULL;
            safe_snprintf(result.error_msg, sizeof(result.error_msg),
                    ERROR_MSG_LAYOUT_CONVERSION ": from %d to %d", src_layout, dst_layout);
            return result;
        }
    }

    if (need_layout_conversion) {
        // Layout conversion needed
        if (src_layout == LAYOUT_NHWC && dst_layout == LAYOUT_NCHW) {
            // NHWC -> NCHW
            memcpy(result.shape.dims, dims, num_dims * sizeof(int32_t));
            // Convert dimension order: [N,H,W,C] -> [N,C,H,W]
            // Keep N at index 0, move C to index 1, H to index 2, W to index 3
            result.shape.dims[1] = dims[3]; // C (was at index 3)
            result.shape.dims[2] = dims[1]; // H (was at index 1)
            result.shape.dims[3] = dims[2]; // W (was at index 2)

            if (!convert_nhwc_to_nchw(tflite_data, result.data,
                                     dims[0], dims[1], dims[2], dims[3],
                                     element_size)) {
                free(result.data);
                free(result.shape.dims);
                result.data = NULL;
                result.shape.dims = NULL;
                safe_snprintf(result.error_msg, sizeof(result.error_msg),
                        ERROR_MSG_LAYOUT_CONVERSION);
                return result;
            }
        } else if (src_layout == LAYOUT_NCHW && dst_layout == LAYOUT_NHWC) {
            // NCHW -> NHWC
            memcpy(result.shape.dims, dims, num_dims * sizeof(int32_t));
            // Convert dimension order: [N,C,H,W] -> [N,H,W,C]
            // Keep N at index 0, move H to index 1, W to index 2, C to index 3
            result.shape.dims[1] = dims[2]; // H (was at index 2)
            result.shape.dims[2] = dims[3]; // W (was at index 3)
            result.shape.dims[3] = dims[1]; // C (was at index 1)

            if (!convert_nchw_to_nhwc(tflite_data, result.data,
                                     dims[0], dims[1], dims[2], dims[3],
                                     element_size)) {
                free(result.data);
                free(result.shape.dims);
                result.data = NULL;
                result.shape.dims = NULL;
                safe_snprintf(result.error_msg, sizeof(result.error_msg),
                        ERROR_MSG_LAYOUT_CONVERSION);
                return result;
            }
        }
    } else {
        // No layout conversion needed, copy directly
        memcpy(result.shape.dims, dims, num_dims * sizeof(int32_t));
        if (!copy_tensor_data(tflite_data, result.data, element_size, total_elements)) {
            free(result.data);
            free(result.shape.dims);
            result.data = NULL;
            result.shape.dims = NULL;
            safe_snprintf(result.error_msg, sizeof(result.error_msg),
                    ERROR_MSG_DATA_COPY);
            return result;
        }
    }

    result.success = true;
    return result;
}

// ============================================================================
// Function implementations
// ============================================================================

static inline size_t get_data_type_size(tensor_data_type_t data_type) {
    switch (data_type) {
        case TENSOR_FLOAT32:
            return sizeof(float);
        case TENSOR_INT32:
            return sizeof(int32_t);
        case TENSOR_UINT8:
            return sizeof(uint8_t);
        case TENSOR_INT64:
            return sizeof(int64_t);
        case TENSOR_INT16:
            return sizeof(int16_t);
        case TENSOR_INT8:
            return sizeof(int8_t);
        case TENSOR_FLOAT16:
            // FLOAT16 is typically 2 bytes, but check for platform support
            #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
                // C23 supports _Float16
                return sizeof(_Float16);
            #elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                // ARM with FP16 support
                return 2;
            #elif defined(__AVX2__) && defined(__F16C__)
                // x86 with F16C support
                return 2;
            #else
                // Generic case - assume 2 bytes
                return 2;
            #endif
        default:
            return 0; // Unsupported type
    }
}

static inline size_t calculate_total_elements(const int32_t* dims, size_t num_dims) {
    if (!dims || num_dims == 0) {
        return 0;
    }

    size_t total = 1;
    for (size_t i = 0; i < num_dims; i++) {
        if (dims[i] <= 0) {
            return 0; // Invalid dimension
        }
        // Check for overflow before multiplication
        if (total > SIZE_MAX / (size_t)dims[i]) {
            return 0; // Overflow would occur
        }
        total *= (size_t)dims[i];
    }
    return total;
}

static inline bool validate_tensor_shape(const int32_t* dims, size_t num_dims) {
    if (!dims || num_dims == 0 || num_dims > 8) { // Limit to maximum 8 dimensions
        return false;
    }

    for (size_t i = 0; i < num_dims; i++) {
        if (dims[i] <= 0) {
            return false;
        }
    }
    return true;
}

static inline bool copy_tensor_data(const void* src, void* dst, size_t element_size, size_t total_elements) {
    if (!src || !dst || element_size == 0 || total_elements == 0) {
        return false;
    }

    // Check for overflow in total_bytes calculation
    if (total_elements > SIZE_MAX / element_size) {
        return false; // Overflow would occur
    }

    size_t total_bytes = element_size * total_elements;
    memcpy(dst, src, total_bytes);
    return true;
}

static inline void print_tensor_info(const tensor_shape_t* shape) {
    if (!shape) {
        printf("Invalid tensor shape: null pointer\n");
        return;
    }
    if (!shape->dims) {
        printf("Invalid tensor shape: null dimensions array\n");
        return;
    }
    printf("Tensor Info:\n");
    printf("  Data Type: %d\n", shape->data_type);
    printf("  Layout: %d\n", shape->layout);
    printf("  Dimensions: %zu [", (size_t)shape->num_dims);
    for (size_t i = 0; i < shape->num_dims; i++) {
        printf("%d", shape->dims[i]);
        if (i < shape->num_dims - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    printf("  Total Elements: %zu\n", shape->total_elements);
    printf("  Element Size: %zu bytes\n", get_data_type_size(shape->data_type));
    // Check for overflow in total size calculation
    size_t element_size = get_data_type_size(shape->data_type);
    if (element_size > 0 && shape->total_elements <= SIZE_MAX / element_size) {
        printf("  Total Size: %zu bytes\n", shape->total_elements * element_size);
    } else {
        printf("  Total Size: overflow or invalid\n");
    }
}

static inline void free_conversion_result(conversion_result_t* result) {
    if (!result) {
        return;
    }
    if (result->data) {
        free(result->data);
        result->data = NULL;
    }
    if (result->shape.dims) {
        free(result->shape.dims);
        result->shape.dims = NULL;
    }
    result->data_size = 0;
    result->shape.num_dims = 0;
    result->shape.total_elements = 0;
    result->success = false;
    memset(result->error_msg, 0, sizeof(result->error_msg));
}



#ifdef __cplusplus
}
#endif

#endif // TENSOR_CONVERTER_H

