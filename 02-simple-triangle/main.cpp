#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <streambuf>
#include <vector>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include "config.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <filesystem>

#include "stb_image_write.h"

#define WIDTH 640
#define HEIGHT 480
#define CHANNELS 4

struct Params {
  uchar4 *image;
  unsigned int image_width;
};

struct RayGenData {
  float r, g, b, a;
};

template<typename T>
struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int> MissSbtRecord;


void loadFile(const std::filesystem::path &filePath, std::string &out) {
  std::ifstream t(filePath);
  out = std::string(
    std::istreambuf_iterator<char>(t),
    std::istreambuf_iterator<char>()
  );
}

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
    << message << "\n";
}

void createContext(OptixDeviceContext &context) {
  // Initialize CUDA
  cudaFree(0);

  CUcontext cuCtx = 0; // zero means take the current context
  optixInit();
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &context_log_cb;
  options.logCallbackLevel = 4;
  optixDeviceContextCreate(cuCtx, &options, &context);
}

void createModule(
  OptixDeviceContext &context,
  OptixModule &module,
  OptixPipelineCompileOptions &pipelineCompileOptions
) {
  OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
  pipelineCompileOptions.usesMotionBlur = false;
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pipelineCompileOptions.numPayloadValues = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

  std::string inputFile;
  loadFile(std::filesystem::path(OPTIX_PROGRAMS_DIR) / "solid_color.optixir", inputFile);

  optixModuleCreate(
    context,
    &module_compile_options,
    &pipelineCompileOptions,
    inputFile.c_str(),
    inputFile.size(),
    nullptr, nullptr,
    &module
  );
}


int main(int argc, char *argv[]) {
  std::cout << "ENTERED MAIN\n";

  //
  // Initialize CUDA and create OptiX context
  //
  OptixDeviceContext context = nullptr;
  createContext(context);

  std::cout << "CREATED CONTEXT\n";

  //
  // Create module
  //
  OptixModule module = nullptr;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  createModule(context, module, pipeline_compile_options);

  std::cout << "CREATED MODULE\n";

  //
  // Create program groups, including NULL miss and hitgroups
  //
  OptixProgramGroup raygen_prog_group = nullptr;
  OptixProgramGroup miss_prog_group = nullptr; {
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
    optixProgramGroupCreate(
      context,
      &raygen_prog_group_desc,
      1, // num program groups
      &program_group_options,
      nullptr, nullptr,
      &raygen_prog_group
    );

    // Leave miss group's module and entryfunc name null
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    optixProgramGroupCreate(
      context,
      &miss_prog_group_desc,
      1, // num program groups
      &program_group_options,
      nullptr, nullptr,
      &miss_prog_group
    );
  }

  //
  // Link pipeline
  //
  OptixPipeline pipeline = nullptr; {
    const uint32_t max_trace_depth = 0;
    OptixProgramGroup program_groups[] = {raygen_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    optixPipelineCreate(
      context,
      &pipeline_compile_options,
      &pipeline_link_options,
      program_groups,
      sizeof(program_groups) / sizeof(program_groups[0]),
      nullptr, nullptr,
      &pipeline
    );

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group: program_groups) {
      optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline);
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    optixUtilComputeStackSizes(
      &stack_sizes, max_trace_depth,
      0, // maxCCDepth
      0, // maxDCDEpth
      &direct_callable_stack_size_from_traversal,
      &direct_callable_stack_size_from_state, &continuation_stack_size
    );
    optixPipelineSetStackSize(
      pipeline, direct_callable_stack_size_from_traversal,
      direct_callable_stack_size_from_state, continuation_stack_size,
      2 // maxTraversableDepth
    );
  }

  //
  // Set up shader binding table
  //
  OptixShaderBindingTable sbt = {}; {
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size);
    RayGenSbtRecord rg_sbt;
    optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt);
    rg_sbt.data = {0.462f, 0.725f, 0.f, 1.0f};
    cudaMemcpy(
      reinterpret_cast<void *>(raygen_record),
      &rg_sbt,
      raygen_record_size,
      cudaMemcpyHostToDevice
    );

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size);
    RayGenSbtRecord ms_sbt;
    optixSbtRecordPackHeader(miss_prog_group, &ms_sbt);
    cudaMemcpy(
      reinterpret_cast<void *>(miss_record),
      &ms_sbt,
      miss_record_size,
      cudaMemcpyHostToDevice
    );

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
  }

  //
  // launch
  //
  uchar4 *device_pixels = nullptr;
  cudaFree(reinterpret_cast<void *>(device_pixels));
  cudaMalloc(
    reinterpret_cast<void **>(&device_pixels),
    WIDTH * HEIGHT * sizeof(uchar4)
  );

  std::vector<uchar4> host_pixels;
  host_pixels.reserve(WIDTH * HEIGHT);

  Params params; {
    CUstream stream;
    cudaStreamCreate(&stream);

    params.image = device_pixels;
    params.image_width = WIDTH;

    CUdeviceptr d_param;
    cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params));
    cudaMemcpy(
      reinterpret_cast<void *>(d_param),
      &params,
      sizeof(params),
      cudaMemcpyHostToDevice
    );

    optixLaunch(pipeline, stream, d_param, sizeof(params), &sbt, WIDTH, HEIGHT, /*depth=*/1);
    // CUDA_SYNC_CHECK();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
      throw "KERNEL EXECUTION FAILED";

    cudaMemcpy(
      static_cast<void *>(host_pixels.data()),
      device_pixels,
      WIDTH * HEIGHT * sizeof(uchar4),
      cudaMemcpyDeviceToHost
    );

    cudaFree(reinterpret_cast<void *>(d_param));
  }

  stbi_write_png(
    "./output.png",
    WIDTH,
    HEIGHT,
    CHANNELS,
    host_pixels.data(),
    WIDTH * sizeof(uchar4)
  );

  //
  // Cleanup
  //
  {
    cudaFree(reinterpret_cast<void *>(sbt.raygenRecord));
    cudaFree(reinterpret_cast<void *>(sbt.missRecordBase));

    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(miss_prog_group);
    optixProgramGroupDestroy(raygen_prog_group);
    optixModuleDestroy(module);

    optixDeviceContextDestroy(context);
  }

  return 0;
}
