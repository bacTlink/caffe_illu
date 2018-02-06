#include <cstdio>
#include <vector>
#include <cmath>

#include "boost/random.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif  // USE_OPENCV

#include "illu/illu_raw_data.hpp"

using namespace illu;
using std::vector;
using std::cerr;
using std::endl;
using namespace cv;

DEFINE_int32(N, 10, "Number of cases");
DEFINE_int32(height, 100, "Height of image");
DEFINE_int32(width, 100, "Width of image");
DEFINE_double(min_light_src_height, 1.0, "Lower bound of height of light source");
DEFINE_double(max_light_src_height, 50.0, "Upper bound of height of light source");
DEFINE_string(dst_datafile, "illu_raw_data.txt", "Destination datafile");
DEFINE_string(dst_dir, "data/illu/", "Destination directory");

namespace {

inline double sqr(double x) { return x * x; }

double random_real_lr(double l, double r) {
  static boost::mt19937 rng;
  boost::uniform_real<> real(l, r);
  return real(rng);
}

double random_int_lr(int l, int r) {
  static boost::mt19937 rng;
  boost::uniform_int<> ranint(l, r);
  return ranint(rng);
}

double random_degree(int i, double h_degree, int H) {
	return (i * 2 * h_degree / H) - h_degree + random_real_lr(0, 2 * h_degree / H);
}

double get_pos(double degree, double height, int H) {
	return tan(degree) * height + H * 0.5;
}

void getImg(int H, int W, double min_height, double max_height,
    PixelRecord* pImg, Mat *pResImg, vector<PhotonRecord> *photons) {
  // Init 
  photons->clear();
	for (int i = 0; i < H; ++i) {
		for (int j = 0; j < W; ++j) {
			const int index = i * W + j;
			PixelRecord &pixel_record = pImg[index];
			pixel_record.photons.clear();
			pixel_record.BRDF_.b_ = pixel_record.BRDF_.g_ = pixel_record.BRDF_.r_ = 1.0;
			pixel_record.pos_.x_ = i - H * 0.5 + 0.5;
			pixel_record.pos_.y_ = j - W * 0.5 + 0.5;
			pixel_record.pos_.z_ = 0;
			pixel_record.reflection_ = 0;
			pixel_record.refraction_ = 0;
			pixel_record.depth_ = sqrt(sqr(pixel_record.pos_.x_) + sqr(pixel_record.pos_.y_) + sqr(min_height * 0.8));
		}
	}

  // Emit photons
  double height = random_real_lr(min_height, max_height);
  double h_degree = atan2(H * 0.5, height);
  double w_degree = atan2(W * 0.5, height);
  for (int t = 0; t < 2; ++t) {
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        PhotonRecord photon;
				double i_degree = random_degree(i, h_degree, H);
				double j_degree = random_degree(j, w_degree, W);
				double i_pos = get_pos(i_degree, height, H);
				double j_pos = get_pos(j_degree, height, W);
				photon.pos_.x_ = i_pos;
				photon.pos_.y_ = j_pos;
				photon.pos_.z_ = 0;
				photon.rgb_.b_ = photon.rgb_.g_ = photon.rgb_.r_ = (2.0 * 2.0 * h_degree * w_degree) / (2.0 * H * W);
				photon.reflection_ = photon.refraction_ = 0;
				photon.depth_ = sqrt(sqr(height) + sqr(i_pos) + sqr(j_pos));
				int num = photons->size();
				int x = round(i_pos), y = round(j_pos);
				for (int dx = -4; dx <= 4; ++dx)
					for (int dy = -4; dy <= 4; ++dy)
						if (0 <= dx + x && dx + x < W
								&& 0 <= dy + y && dy + y < H)
						pImg[(dx + x) * W + dy + y].photons.push_back(num);
				photons->push_back(photon);
      }
    }
  }

  // Target image
  for (int i = 0; i < H; ++i) {
    uchar* mat_i = pResImg->ptr<uchar>(i);
    for (int j = 0; j < W; ++j) {
      double x = fabs(i - H * 0.5);
      double y = fabs(j - W * 0.5);
      double col = 255;
      col *= sqr(height) / ((sqr(x) + sqr(height)));
      col *= sqr(height) / ((sqr(y) + sqr(height)));
      for (int k = 0; k < 3; ++k)
        mat_i[j * 3 + k] = (uchar)col;
    }
  }

  // Add baffle
  int lx, ly, ux, uy;
  lx = random_int_lr(0, H - 1);
  ux = random_int_lr(0, H - 1);
  if (lx > ux)
    std::swap(lx, ux);
  ly = random_int_lr(0, W - 1);
  uy = random_int_lr(0, W - 1);
  if (ly > uy)
    std::swap(ly, uy);
  for (int i = lx; i <= ux; ++i) {
    uchar* mat_i = pResImg->ptr<uchar>(i);
    for (int j = ly; j <= uy; ++j) {
			const int index = i * W + j;
      pImg[index].photons.clear();
      for (int k = 0; k < 3; ++k)
        mat_i[j * 3 + k] = 0;
    }
  }
}

} // namespace

int main(int argc, char **argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Generate primary raw data for illu\n"
        "Usage:\n"
        "    illu_gen_rawdata [FLAGS]\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/illu_gen_rawdata");
    return 1;
  }

  FILE* fout = fopen((FLAGS_dst_dir + FLAGS_dst_datafile).c_str(), "w");
  int N = FLAGS_N;
  const int H = FLAGS_height;
  const int W = FLAGS_width;
  fprintf(fout, "%d\n", N);
  PixelRecord *pImg = new PixelRecord[H * W];
  Mat *pResImg = new Mat(H, W, CV_8UC3, Scalar(0, 0, 0));
  vector<PhotonRecord> *photons = new vector<PhotonRecord>();
  for (int ca = 0; ca < N; ++ca) {
    std::cerr << "Img" << ca << std::endl;
    getImg(H, W, FLAGS_min_light_src_height, FLAGS_max_light_src_height,
      pImg, pResImg, photons);
    char bmp_file[100];
    sprintf(bmp_file, "%sraw_%d.bmp", FLAGS_dst_dir.c_str(), ca);
    imwrite(bmp_file, *pResImg);
    fprintf(fout, "%s\n", bmp_file);
    fprintf(fout, "%lu\n", photons->size());
    for (vector<PhotonRecord>::iterator i = photons->begin(); i != photons->end(); ++i) {
      fprintf(fout, "%.6f %.6f %.6f %.6f %.6f %.6f %d %d %.6f\n",
          i->pos_.x_, i->pos_.y_, i->pos_.z_,
          i->rgb_.b_, i->rgb_.g_, i->rgb_.r_,
          i->reflection_, i->reflection_, i->depth_);
    }
    fprintf(fout, "%d %d\n", H, W);
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j) {
        const int index = i * W + j;
        PixelRecord &pr = pImg[index];
        fprintf(fout, "%.6f %.6f %.6f %.6f %.6f %.6f %d %d %.6f ",
            pr.pos_.x_, pr.pos_.y_, pr.pos_.z_,
            pr.BRDF_.b_, pr.BRDF_.g_, pr.BRDF_.r_,
            pr.reflection_, pr.refraction_, pr.depth_);
        fprintf(fout, "%lu", pr.photons.size());
        for (vector<int>::iterator x = pr.photons.begin(); x != pr.photons.end(); ++x)
          fprintf(fout, " %d", *x);
        fprintf(fout, "\n");
      }
  }
  delete [] pImg;
  delete pResImg;
  delete photons;
  fclose(fout);

#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
