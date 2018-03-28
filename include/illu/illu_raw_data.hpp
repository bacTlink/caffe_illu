#ifndef ILLU_RAW_DATA_H_
#define ILLU_RAW_DATA_H_

#include <vector>

namespace illu {

struct Pos {
  double x_, y_, z_;
  Pos(double x = 0, double y = 0, double z = 0):
    x_(x), y_(y), z_(z) { }
};

struct RGB {
  double r_, g_, b_;
  RGB(double r = 0, double g = 0, double b = 0):
    r_(r), g_(g), b_(b) { }
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
