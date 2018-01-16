// This program converts illu raw data format to lmdb
// as Datum proto buffers.
// Usage:
//   illu_to_lmdb [FLAGS] raw_file
//
// where raw_file is the data of illu defined in
// illu/raw_data_format.
//   ....

#include <cstdio>

#include "opencv2/opencv.hpp"

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_string(prefix, "", "The prefix of output");
DEFINE_string(dst_dir, "./", "Destination directory");
DEFINE_int32(photon_per_pixel, 100, "Remain photon count per pixel");

struct Pos {
  float x_, y_, z_;
};

struct RGB {
  float b_, g_, r_;
};

struct PhotonRecord {
  char reflection_;
  char refraction_;
  RGB rgb_;
  float depth_;
  Pos pos_;
};

vector<PhotonRecord> photons;
vector<int> id;

inline float sqr(const float& x) { return x * x; }
float squ_dis(const Pos& p1, const Pos& p2) {
  return sqr(p1.x_ - p2.x_) + sqr(p1.y_ - p2.y_) + sqr(p1.z_ - p2.z_);
}

struct Comparator {
	Comparator(Pos p): p_(p) {}
	bool operator () (const int& id1, const int& id2) {
		return squ_dis(photons[id1].pos_, p_) < squ_dis(photons[id2].pos_, p_);
	}
	Pos p_;
};

void init_datum(Datum *datum, const int &chas, const int &H, const int &W) {
  datum->set_channels(chas);
  datum->set_height(H);
  datum->set_width(W);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert ill raw data format to lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    illu_to_lmdb [FLAGS] ILLU_DATA_FILE\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/illu_to_lmdb");
    return 1;
  }

  scoped_ptr<db::DB> db_conv(db::GetDB("lmdb"));
  db_conv->Open(FLAGS_dst_dir + FLAGS_prefix + "_conv", db::NEW);
  scoped_ptr<db::Transaction> txn_conv(db_conv->NewTransaction());

  scoped_ptr<db::DB> db_BRDF(db::GetDB("lmdb"));
  db_BRDF->Open(FLAGS_dst_dir + FLAGS_prefix + "_BRDF", db::NEW);
  scoped_ptr<db::Transaction> txn_BRDF(db_BRDF->NewTransaction());

  scoped_ptr<db::DB> db_rrd(db::GetDB("lmdb"));
  db_rrd->Open(FLAGS_dst_dir + FLAGS_prefix + "_rrd", db::NEW);
  scoped_ptr<db::Transaction> txn_rrd(db_rrd->NewTransaction());

  scoped_ptr<db::DB> db_photon(db::GetDB("lmdb"));
  db_photon->Open(FLAGS_dst_dir + FLAGS_prefix + "_photon", db::NEW);
  scoped_ptr<db::Transaction> txn_photon(db_photon->NewTransaction());

  int N, P, H, W;
  char filename[100];
  int preH = -1, preW = -1;
  Datum datum;
  string out;

  FILE* fin = fopen(argv[1], "r");
  CHECK_EQ(fscanf(fin, "%d", &N), 1);
  for (int bat = 0; bat < N; ++bat) {
    CHECK_EQ(fscanf(fin, "%s", filename), 1);
    cv::Mat conv = cv::imread(filename);
    CHECK(conv.data);
    CHECK_EQ(conv.channels(), 3);
    vector<cv::Mat> channels(3);
    cv::split(conv, channels);
    for (int c = 0; c < 3; ++c) {
      CVMatToDatum(channels[c], &datum);
      CHECK(datum.SerializeToString(&out));
      txn_conv->Put(filename, out);
    }

    CHECK_EQ(fscanf(fin, "%d", &P), 1);
    photons.clear();
    PhotonRecord photon;
    for (int i = 0; i < P; ++i) {
      int tmp_refl, tmp_refr;
      CHECK_EQ(fscanf(fin, "%f%f%f%f%f%f%d%d%f",
            &photon.pos_.x_, &photon.pos_.y_, &photon.pos_.z_,
            &photon.rgb_.b_, &photon.rgb_.g_, &photon.rgb_.r_,
            &tmp_refl, &tmp_refr, &photon.depth_),
          9);
      photon.reflection_ = static_cast<char>(tmp_refl);
      photon.refraction_ = static_cast<char>(tmp_refr);
      photons.push_back(photon);
    }

    CHECK_EQ(fscanf(fin, "%d%d", &H, &W), 2);
    CHECK_EQ(H, conv.rows);
    CHECK_EQ(W, conv.cols);

    if (preH != -1 || preW != -1) {
      CHECK_EQ(H, preH);
      CHECK_EQ(W, preW);
    } else {
      preH = H;
      preW = W;
    }

    Datum BRDF_mat[3], rrd_mat, photon_mat[3];
    init_datum(&rrd_mat, 3, H, W);
    for (int i = 0; i < 3; ++i) {
      init_datum(BRDF_mat + i, 1, H, W);
      init_datum(photon_mat + i, 5 * FLAGS_photon_per_pixel, H, W);
    }
    RGB BRDF_color;
		Pos pos;
    float reflection, refraction;
    float depth;
    int M;
		for (int i = 0; i < H; ++i)
			for (int j = 0; j < W; ++j) {
        const int index = i * W + j;
        CHECK_EQ(fscanf(fin, "%f%f%f",
              &BRDF_color.b_, &BRDF_color.g_, &BRDF_color.r_), 3);
        BRDF_mat[0].set_float_data(index, BRDF_color.b_);
        BRDF_mat[1].set_float_data(index, BRDF_color.g_);
        BRDF_mat[2].set_float_data(index, BRDF_color.r_);
				CHECK_EQ(fscanf(fin, "%f%f%f", &pos.x_, &pos.y_, &pos.z_), 3);
        CHECK_EQ(fscanf(fin, "%f%f%f", &reflection, &refraction, &depth), 3);
        rrd_mat.set_float_data(0 * H * W + index, reflection);
        rrd_mat.set_float_data(1 * H * W + index, refraction);
        rrd_mat.set_float_data(2 * H * W + index, depth);
				CHECK_EQ(fscanf(fin, "%d", &M), 1);
        id.clear();
				for (int p = 0; p < M; ++p) {
          int tmp_id;
					CHECK_EQ(fscanf(fin, "%d", &tmp_id), 1);
          id.push_back(tmp_id);
        }
				sort(id.begin(), id.end(), Comparator(pos));
        for (int p = 0; i < FLAGS_photon_per_pixel; ++i) {
          for (int c = 0; c < 3; ++c) {
            photon_mat[c].set_float_data((p * 5 + 0) * H * W + index, squ_dis(pos, photons[p].pos_));
            photon_mat[c].set_float_data((p * 5 + 2) * H * W + index, photons[p].reflection_);
            photon_mat[c].set_float_data((p * 5 + 2) * H * W + index, photons[p].refraction_);
            photon_mat[c].set_float_data((p * 5 + 4) * H * W + index, photons[p].depth_);
          }
          photon_mat[0].set_float_data((p * 5 + 1) * H * W + index, photons[p].rgb_.b_);
          photon_mat[1].set_float_data((p * 5 + 1) * H * W + index, photons[p].rgb_.g_);
          photon_mat[2].set_float_data((p * 5 + 1) * H * W + index, photons[p].rgb_.r_);
        }
			}
    for (int c = 0; c < 3; ++c) {
      CHECK(BRDF_mat[c].SerializeToString(&out));
      txn_BRDF->Put(filename, out);
    }
    CHECK(rrd_mat.SerializeToString(&out));
    for (int c = 0; c < 3; ++c) {
      txn_rrd->Put(filename, out);
    }
    for (int c = 0; c < 3; ++c) {
      CHECK(photon_mat[c].SerializeToString(&out));
      txn_photon->Put(filename, out);
    }
  }

  fclose(fin);

#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
