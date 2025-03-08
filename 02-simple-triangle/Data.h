//
// Created by vector on 3/7/25.
//

#ifndef DATA_H
#define DATA_H

struct Params {
  uchar4 *image;
  unsigned int imageWidth;
  unsigned int imageHeight;
  float3 cameraEye, cameraU, cameraV, cameraW;
  OptixTraversableHandle traversableHandle;
};

struct RayGenData {
};


struct HitGroupData {
};


struct MissData {
  float3 bgColor;
};

#endif //DATA_H
