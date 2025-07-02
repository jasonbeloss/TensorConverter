# TensorConverter

A lightweight C header for safe and efficient tensor data layout and type conversion between ONNX and TFLite formats.

**Author**: Jiacheng.Du

## Features
- 支持 ONNX (NCHW) 和 TFLite (NHWC) 张量布局转换
- 支持多种数据类型：float32, int32, uint8, int64, int16, int8, float16
- 提供内存安全检查（溢出、空指针、分配失败）
- 所有 API 和类型使用 snake_case 命名规范
- 纯 C99 实现，仅头文件，跨平台兼容

## Data Types
```c
// Data type enum
typedef enum {
    TENSOR_FLOAT32 = 0,
    TENSOR_INT32 = 1,
    TENSOR_UINT8 = 2,
    TENSOR_INT64 = 3,
    TENSOR_INT16 = 6,
    TENSOR_INT8 = 8,
    TENSOR_FLOAT16 = 9
} tensor_data_type_t;
```

## Layout Types
```c
typedef enum {
    LAYOUT_UNKNOWN = 0,
    LAYOUT_NCHW = 1,    // ONNX: [N, C, H, W]
    LAYOUT_NHWC = 2,    // TFLite: [N, H, W, C]
    LAYOUT_GENERIC = 3
} tensor_layout_t;
```

## Main Structures
```c
typedef struct {
    int32_t* dims;
    size_t num_dims;
    tensor_data_type_t data_type;
    size_t total_elements;
    tensor_layout_t layout;
} tensor_shape_t;

typedef struct {
    void* data;
    size_t data_size;
    tensor_shape_t shape;
    bool success;
    char error_msg[256];
} conversion_result_t;
```

## Main API

### Conversion with Layout Support
```c
conversion_result_t onnx_to_tflite_with_layout(
    const void* onnx_data,
    const int32_t* dims,
    size_t num_dims,
    tensor_data_type_t data_type,
    tensor_layout_t src_layout,
    tensor_layout_t dst_layout);

conversion_result_t tflite_to_onnx_with_layout(
    const void* tflite_data,
    const int32_t* dims,
    size_t num_dims,
    tensor_data_type_t data_type,
    tensor_layout_t src_layout,
    tensor_layout_t dst_layout);
```

### Utilities
```c
void free_conversion_result(conversion_result_t* result);
void print_tensor_info(const tensor_shape_t* shape);
size_t get_data_type_size(tensor_data_type_t data_type);
size_t calculate_total_elements(const int32_t* dims, size_t num_dims);
bool validate_tensor_shape(const int32_t* dims, size_t num_dims);
```

## Usage Example
```c
#include "tensor_converter.h"

// Example: Convert ONNX NCHW float32 tensor to TFLite NHWC
float onnx_data[1*3*224*224];
int32_t dims[4] = {1, 3, 224, 224};

// Convert with explicit layout conversion: NCHW -> NHWC
conversion_result_t result = onnx_to_tflite_with_layout(
    onnx_data, dims, 4, TENSOR_FLOAT32, LAYOUT_NCHW, LAYOUT_NHWC);

if (result.success) {
    printf("转换成功！\n");
    print_tensor_info(&result.shape);
    // Use result.data ...
    free_conversion_result(&result);
} else {
    printf("转换失败: %s\n", result.error_msg);
}
```

## Safety Notes
- 所有内存分配都会检查失败情况
- 所有索引和大小计算都会检查溢出
- 所有指针参数都会检查空指针
- 使用 `free_conversion_result` 释放 `conversion_result_t` 中的内存
- 调用者负责提供有效的输入数据和维度

## 重要变更
- 已删除基础的 `onnx_to_tflite` 和 `tflite_to_onnx` 函数（仅进行内存拷贝）
- 现在只提供带布局转换功能的 `*_with_layout` 版本，提供更完整的转换能力

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License allows you to:
- ✅ Use the code commercially
- ✅ Modify and distribute
- ✅ Include in proprietary software
- ✅ Private use

Requirements:
- 📄 Include the original copyright notice
- 📄 Include the license text 