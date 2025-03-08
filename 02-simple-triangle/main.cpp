#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sutil/sutil.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Data.h"
#include "config.h"
#include "sutil/Exception.h"

#define WIDTH 640
#define HEIGHT 480
#define CHANNELS 4

template<typename T>
struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct GlobalAccelerationStructure {
  OptixTraversableHandle handle;
  CUdeviceptr            outputBuffer;
};

struct ProgramGroups {
  OptixProgramGroup raygen;
  OptixProgramGroup miss;
  OptixProgramGroup hit;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

std::string LoadFileString(const std::filesystem::path &filePath) {
  std::ifstream t(filePath);
  return {std::istreambuf_iterator(t), std::istreambuf_iterator<char>()};
}

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

OptixDeviceContext CreateContext() {
  // NRVO
  // Initialize CUDA
  CUDA_CHECK(cudaFree(nullptr));

  OPTIX_CHECK(optixInit());
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction       = &context_log_cb;
  options.logCallbackLevel          = 4;
#if defined(DEBUG)
  // This may incur significant performance cost and should only be done during
  // development.
  options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
  CUcontext cuCtx = nullptr; // zero means take the current context

  OptixDeviceContext context;
  optixDeviceContextCreate(cuCtx, &options, &context);
  return context;
}

GlobalAccelerationStructure BuildAccelerationStructures( // NVRO
  OptixDeviceContext context) {
  GlobalAccelerationStructure gas{};

  // Use default options for simplicity.  In a real use case we would want to
  // enable compaction, etc
  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
  accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

  // Triangle build input: simple list of three vertices
  const std::array<float3, 3> vertices = {{{-0.5f, -0.5f, 0.0f}, {0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 0.0f}}};

  CUdeviceptr dVertices = 0;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dVertices), sizeof(float3) * vertices.size()));
  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void *>(dVertices), vertices.data(), sizeof(float3) * vertices.size(), cudaMemcpyHostToDevice));

  // Our build input is a simple list of non-indexed triangle vertices
  const uint32_t  triangle_input_flags[1]    = {OPTIX_GEOMETRY_FLAG_NONE};
  OptixBuildInput triangle_input             = {};
  triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.numVertices   = static_cast<uint32_t>(vertices.size());
  triangle_input.triangleArray.vertexBuffers = &dVertices;
  triangle_input.triangleArray.flags         = triangle_input_flags;
  triangle_input.triangleArray.numSbtRecords = 1;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
    context, &accel_options, &triangle_input,
    1, // Number of build inputs
    &gas_buffer_sizes));
  CUdeviceptr dTempGasBuffer;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTempGasBuffer), gas_buffer_sizes.tempSizeInBytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gas.outputBuffer), gas_buffer_sizes.outputSizeInBytes));

  OPTIX_CHECK(optixAccelBuild(
    context,
    nullptr, // CUDA stream
    &accel_options, &triangle_input,
    1, // num build inputs
    dTempGasBuffer, gas_buffer_sizes.tempSizeInBytes, gas.outputBuffer, gas_buffer_sizes.outputSizeInBytes, &gas.handle,
    nullptr, // emitted property list
    0        // num emitted properties
    ));

  // We can now free the scratch space buffer used during build and the vertex
  // inputs, since they are not needed by our trivial shading method
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(dTempGasBuffer)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(dVertices)));

  return gas;
}

OptixModule CreateModule(
  const OptixDeviceContext &context, const OptixModuleCompileOptions &moduleCompileOptions,
  const OptixPipelineCompileOptions &pipelineCompileOptions) {
  std::string inputFile = LoadFileString(std::filesystem::path(OPTIX_PROGRAMS_DIR) / "triangle.optixir");

  OptixModule module;
  OPTIX_CHECK(optixModuleCreate(
    context, &moduleCompileOptions, &pipelineCompileOptions, inputFile.c_str(), inputFile.size(), nullptr, nullptr,
    &module));

  return module;
}

ProgramGroups CreateProgramGroups(OptixDeviceContext context, OptixModule module) {
  ProgramGroups            pg{};
  OptixProgramGroupOptions programGroupOptions = {}; // Initialize to zeros

  OptixProgramGroupDesc pgDescRaygen    = {}; //
  pgDescRaygen.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDescRaygen.raygen.module            = module;
  pgDescRaygen.raygen.entryFunctionName = "__raygen__rg";
  OPTIX_CHECK(optixProgramGroupCreate(
    context, &pgDescRaygen,
    1, // num program groups
    &programGroupOptions, nullptr, nullptr, &pg.raygen));

  // Leave miss group's module and entryfunc name null
  OptixProgramGroupDesc pgDescMiss  = {};
  pgDescMiss.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDescMiss.miss.module            = module;
  pgDescMiss.miss.entryFunctionName = "__miss__ms";
  OPTIX_CHECK(optixProgramGroupCreate(
    context, &pgDescMiss,
    1, // num program groups
    &programGroupOptions, nullptr, nullptr, &pg.miss));

  OptixProgramGroupDesc pgDescHit        = {};
  pgDescHit.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgDescHit.hitgroup.moduleCH            = module;
  pgDescHit.hitgroup.entryFunctionNameCH = "__closesthit__ch";
  OPTIX_CHECK(optixProgramGroupCreate(
    context, &pgDescHit,
    1, // num program groups
    &programGroupOptions, nullptr, nullptr, &pg.hit));

  return pg;
}

void CreatePipeline(
  OptixDeviceContext context, OptixPipelineCompileOptions pipelineCompileOptions, OptixPipeline &pipeline,
  ProgramGroups &pg) {
  constexpr uint32_t max_trace_depth  = 1;
  OptixProgramGroup  program_groups[] = {pg.raygen, pg.miss, pg.hit};

  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth            = max_trace_depth;
  OPTIX_CHECK(optixPipelineCreate(
    context, &pipelineCompileOptions, &pipeline_link_options, program_groups, std::size(program_groups), nullptr,
    nullptr, &pipeline));

  OptixStackSizes stack_sizes = {};
  for (auto &prog_group: program_groups) {
    OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
  }

  uint32_t direct_callable_stack_size_from_traversal;
  uint32_t direct_callable_stack_size_from_state;
  uint32_t continuation_stack_size;
  OPTIX_CHECK(optixUtilComputeStackSizes(
    &stack_sizes, max_trace_depth,
    0, // maxCCDepth
    0, // maxDCDEpth
    &direct_callable_stack_size_from_traversal, &direct_callable_stack_size_from_state, &continuation_stack_size));
  OPTIX_CHECK(optixPipelineSetStackSize(
    pipeline, direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size,
    1 // maxTraversableDepth
    ));
}

void CreateShaderBindingTable(OptixShaderBindingTable &sbt, ProgramGroups pg) {
  CUdeviceptr recordRaygen;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&recordRaygen), sizeof(RayGenSbtRecord)));
  RayGenSbtRecord sbtRaygen;
  OPTIX_CHECK(optixSbtRecordPackHeader(pg.raygen, &sbtRaygen));
  CUDA_CHECK(
    cudaMemcpy(reinterpret_cast<void *>(recordRaygen), &sbtRaygen, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

  CUdeviceptr recordMiss;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&recordMiss), sizeof(MissSbtRecord)));
  MissSbtRecord sbtMiss;
  sbtMiss.data = {0.3f, 0.3f, 0.3f};
  OPTIX_CHECK(optixSbtRecordPackHeader(pg.miss, &sbtMiss));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(recordMiss), &sbtMiss, sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

  CUdeviceptr recordHit;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&recordHit), sizeof(RayGenSbtRecord)));
  HitGroupSbtRecord sbtHit;
  OPTIX_CHECK(optixSbtRecordPackHeader(pg.hit, &sbtHit));
  CUDA_CHECK(
    cudaMemcpy(reinterpret_cast<void *>(recordHit), &sbtHit, sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

  sbt.raygenRecord                = recordRaygen;
  sbt.missRecordBase              = recordMiss;
  sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
  sbt.missRecordCount             = 1;
  sbt.hitgroupRecordBase          = recordHit;
  sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
  sbt.hitgroupRecordCount         = 1;
}

int main(int argc, char *argv[]) {
  // CONTEXT
  OptixDeviceContext context = CreateContext();

  // ACCELERATION STRUCTURES
  GlobalAccelerationStructure gas = BuildAccelerationStructures(context);

  // COMPILE OPTIONS
  // module compilation options
  OptixModuleCompileOptions moduleCompileOptions = {};
#if defined(DEBUG)
  moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

  // pipeline compilation options
  OptixPipelineCompileOptions pipelineCompileOptions = {};
  pipelineCompileOptions.usesMotionBlur              = false;
  pipelineCompileOptions.traversableGraphFlags       = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.numPayloadValues            = 3;
  pipelineCompileOptions.numAttributeValues          = 3;
  pipelineCompileOptions.exceptionFlags              = OPTIX_EXCEPTION_FLAG_NONE;
  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
  pipelineCompileOptions.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  // MODULE
  OptixModule module = CreateModule(context, moduleCompileOptions, pipelineCompileOptions);

  // PROGRAM GROUPS
  ProgramGroups pg = CreateProgramGroups(context, module);

  // PIPELINE
  OptixPipeline pipeline = nullptr;
  CreatePipeline(context, pipelineCompileOptions, pipeline, pg);

  // SHADER BINDING TABLE
  OptixShaderBindingTable sbt = {};
  CreateShaderBindingTable(sbt, pg);

  // LAUNCH
  uchar4 *device_pixels = nullptr;
  CUDA_CHECK(cudaFree(device_pixels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_pixels), WIDTH * HEIGHT * sizeof(uchar4)));

  std::vector<uchar4> host_pixels;
  host_pixels.reserve(WIDTH * HEIGHT);

  Params params{};

  //
  {
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    params.image             = device_pixels;
    params.imageWidth        = WIDTH;
    params.imageHeight       = HEIGHT;
    params.traversableHandle = gas.handle;
    params.cameraPosition    = {0.0f, 0.0f, -2.0f};
    params.cameraU           = {1.0f, 0.0f, 0.0f};
    params.cameraV           = {0.0f, 1.0f, 0.0f};
    params.cameraW           = {0.0f, 0.0f, 1.0f};

    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_param), &params, sizeof(params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, WIDTH, HEIGHT, /*depth=*/1));
    // CUDA_SYNC_CHECK();

    if (cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess)
      throw R"(KERNEL EXECUTION FAILED)";

    CUDA_CHECK(cudaMemcpy(host_pixels.data(), device_pixels, WIDTH * HEIGHT * sizeof(uchar4), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));
  }

  //
  // Display results
  //
  {
    sutil::ImageBuffer buffer;
    buffer.data         = host_pixels.data();
    buffer.width        = WIDTH;
    buffer.height       = HEIGHT;
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
    sutil::displayBufferWindow(argv[0], buffer);
  }

  // stbi_write_png(
  //   "./output.png",
  //   WIDTH,
  //   HEIGHT,
  //   CHANNELS,
  //   host_pixels.data(),
  //   WIDTH * sizeof(uchar4)
  // );

  //
  // Cleanup
  //
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(gas.outputBuffer)));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(pg.hit));
    OPTIX_CHECK(optixProgramGroupDestroy(pg.miss));
    OPTIX_CHECK(optixProgramGroupDestroy(pg.raygen));
    OPTIX_CHECK(optixModuleDestroy(module));

    OPTIX_CHECK(optixDeviceContextDestroy(context));
  }

  return 0;
}
