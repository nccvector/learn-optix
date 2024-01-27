#include <optix.h>

struct Params {
  uchar4 *image;
  unsigned int image_width;
};

struct RayGenData {
  float r, g, b, a;
};

extern "C" {
__constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color() {
  uint3 launch_index = optixGetLaunchIndex();
  RayGenData *rtData = (RayGenData *) optixGetSbtDataPointer();

  params.image[launch_index.y * params.image_width + launch_index.x] = {
      (unsigned char) (rtData->r * 255),
      (unsigned char) (rtData->g * 255),
      (unsigned char) (rtData->b * 255),
      (unsigned char) (rtData->a * 255)
  };
}
