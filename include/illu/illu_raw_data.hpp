#ifndef ILLU_RAW_DATA_H_
#define ILLU_RAW_DATA_H_

#include <vector>

namespace illu {

struct Pos {
  double x_, y_, z_;
};

struct RGB {
  double b_, g_, r_;
};

struct PhotonRecord {
  char reflection_;
  char refraction_;
  RGB rgb_;
  double depth_;
  Pos pos_;
};

struct PixelRecord {
  char reflection_;
  char refraction_;
  RGB BRDF_;
  Pos pos_;
  double depth_;
  std::vector<int> photons;
};

} // namespace illu

#endif // ILLU_RAW_DATA_H_
