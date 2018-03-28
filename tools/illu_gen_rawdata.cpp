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
DEFINE_int32(height, 224, "Height of image");
DEFINE_int32(width, 224, "Width of image");
DEFINE_double(min_light_src_height, 1.0, "Lower bound of height of light source");
DEFINE_double(max_light_src_height, 50.0, "Upper bound of height of light source");
DEFINE_string(dst_datafile, "illu_raw_data.txt", "Destination datafile");
DEFINE_string(dst_dir, "/data3/lzh/illu/", "Destination directory");
DEFINE_int32(photon_per_pixel, 20, "Remain photon count per pixel");

namespace {

inline double sqr(double x) { return x * x; }
double squ_dis(const Pos& p1, const Pos& p2) {
  return sqr(p1.x_ - p2.x_) + sqr(p1.y_ - p2.y_) + sqr(p1.z_ - p2.z_);
}

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

double get_pos(double degree, double height) {
	return tan(degree) * height;
}

double get_sphere_rectangle_area(double alpha, double beta, double radius) {
  double gamma = acos(cos(alpha) * cos(beta));
  double A = atan2(sin(beta), tan(alpha));
  double B = atan2(sin(alpha), tan(beta));
  double C = acos(sin(A) * sin(B) * cos(gamma) - cos(A) * cos(B));
  return (4 * C - 2 * M_PI) * sqr(radius);
}

void addPhoton(PixelRecord *pImg, vector<PhotonRecord> *photons, const int &index,
    const int &num, Pos hit_pos, Pos num_pos) {
  vector<int> &num_photons = pImg[index].photons;
  double num_dis = squ_dis(hit_pos, num_pos);
  bool inserted = false;
  for (vector<int>::iterator i = num_photons.begin();
      i != num_photons.end(); ++i) {
    if (num_dis <= squ_dis((*photons)[*i].pos_, hit_pos)) {
      num_photons.insert(i, num);
      inserted = true;
      break;
    }
  }
  if (!inserted && num_photons.size() < FLAGS_photon_per_pixel) {
    num_photons.push_back(num);
  } else if (num_photons.size() > FLAGS_photon_per_pixel) {
    num_photons.pop_back();
  }
}

void getImg(int H, int W, double min_height, double max_height,
    PixelRecord *pImg, Mat *pResImg, vector<PhotonRecord> *photons) {
  // Init 
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
  photons->clear();
  double height = random_real_lr(min_height, max_height);
  double h_degree = atan2(H * 0.5, height);
  double w_degree = atan2(W * 0.5, height);
  const int photon_iters = 2;
  double flux_per_photon = get_sphere_rectangle_area(h_degree, w_degree, 1.0)
                            * sqr(height)
                            / (photon_iters * H * W);
  for (int t = 0; t < photon_iters; ++t) {
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        PhotonRecord photon;
				double i_degree = random_degree(i, h_degree, H);
				double j_degree = random_degree(j, w_degree, W);
				double i_pos = get_pos(i_degree, height);
				double j_pos = get_pos(j_degree, height);
				photon.pos_.x_ = i_pos;
				photon.pos_.y_ = j_pos;
				photon.pos_.z_ = 0;
				photon.rgb_.b_ = photon.rgb_.g_ = photon.rgb_.r_ = flux_per_photon;
				photon.reflection_ = photon.refraction_ = 0;
				photon.depth_ = sqrt(sqr(height) + sqr(i_pos) + sqr(j_pos));
				int num = photons->size();
				int x = round(i_pos + H * 0.5), y = round(j_pos + W * 0.5);
				for (int new_x = max(0, x - 16); new_x <= x + 16 && new_x < H; ++new_x)
					for (int new_y = max(0, y - 16); new_y <= y + 16 && new_y < W; ++new_y) {
            const int index = new_x * W + new_y;
            Pos hit_pos(new_x - H * 0.5, new_y - W * 0.5, 0);
            addPhoton(pImg, photons, index, num, hit_pos, photon.pos_);
          }
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
      col *= pow(1 + (sqr(x) + sqr(y)) / sqr(height), -1.5);
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
    char bmp_file[100];
    sprintf(bmp_file, "%sraw_%d.bmp", FLAGS_dst_dir.c_str(), ca);
    LOG(INFO) << bmp_file;
    getImg(H, W, FLAGS_min_light_src_height, FLAGS_max_light_src_height,
      pImg, pResImg, photons);
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
            pr.BRDF_.b_, pr.BRDF_.g_, pr.BRDF_.r_,
            pr.pos_.x_, pr.pos_.y_, pr.pos_.z_,
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
