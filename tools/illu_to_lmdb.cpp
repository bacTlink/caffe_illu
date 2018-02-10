// This program converts illu raw data format to lmdb
// as Datum proto buffers.
// Usage:
//   illu_to_lmdb [FLAGS]
//
// where raw_file is the data of illu defined in
// illu/raw_data_format.
//

#include <cstdio>
#include <cstdlib>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif  // USE_OPENCV

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "illu/illu_raw_data.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::cerr;
using std::endl;
using boost::scoped_ptr;
using namespace illu;

DEFINE_string(prefix, "raw_data", "The prefix of output");
DEFINE_string(dst_dir, "/data3/lzh/illu/", "Destination directory");
DEFINE_string(src_datafile, "/data3/lzh/illu/illu_raw_data.txt", "Source raw datafile");
DEFINE_int32(photon_per_pixel, 20, "Remain photon count per pixel");

vector<PhotonRecord> photons;
vector<int> id;

inline double sqr(const double& x) { return x * x; }
double squ_dis(const Pos& p1, const Pos& p2) {
  return sqr(p1.x_ - p2.x_) + sqr(p1.y_ - p2.y_) + sqr(p1.z_ - p2.z_);
}

struct Comparator {
	Comparator(Pos p): p_(p) {}
	bool operator () (const int& id1, const int& id2) {
		return squ_dis(photons[id1].pos_, p_) < squ_dis(photons[id2].pos_, p_);
	}
	Pos p_;
};

void init_datum(Datum *datum, const int &C, const int &H, const int &W) {
  datum->set_channels(C);
  datum->set_height(H);
  datum->set_width(W);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  for (int x = 0; x < C * H * W; ++x)
    datum->add_float_data(0);
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert illu raw data format to lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    illu_to_lmdb [FLAGS]\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/illu_to_lmdb");
    return 1;
  }

#define DEF_DB(db_name) \
  CHECK_NE(system(("rm -rf " + FLAGS_dst_dir + FLAGS_prefix + "_" + #db_name).c_str()), -1); \
  scoped_ptr<db::DB> db_##db_name(db::GetDB("lmdb")); \
  db_##db_name->Open(FLAGS_dst_dir + FLAGS_prefix + "_" + #db_name, db::NEW); \
  scoped_ptr<db::Transaction> txn_##db_name(db_##db_name->NewTransaction());

  DEF_DB(conv);
  DEF_DB(BRDF);
  DEF_DB(rrd);
  DEF_DB(photon_dis);
  DEF_DB(photon_flux);
  DEF_DB(photon_rrd);

  int N, P, H, W;
  char filename[100];
  Datum BRDF_mat[3], rrd_mat, photon_dis_mat, photon_rrd_mat, photon_flux_mat[3];
  int preH = -1, preW = -1;
  Datum datum;
  string out;

  int commit_cyc = 1;
  FILE* fin = fopen(FLAGS_src_datafile.c_str(), "r");
  CHECK_EQ(fscanf(fin, "%d", &N), 1);
  for (int bat = 0; bat < N; ++bat) {
    LOG(INFO) << "Image: " << bat;

    // Converge Result
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
      CHECK_EQ(fscanf(fin, "%lf%lf%lf%lf%lf%lf%d%d%lf",
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
      // Init matrices
      preH = H;
      preW = W;
      init_datum(&rrd_mat, 3, H, W);
      init_datum(&photon_rrd_mat, 3 * FLAGS_photon_per_pixel, H, W);
      init_datum(&photon_dis_mat, FLAGS_photon_per_pixel, H, W);
      for (int i = 0; i < 3; ++i) {
        init_datum(BRDF_mat + i, 1, H, W);
        init_datum(photon_flux_mat + i, FLAGS_photon_per_pixel, H, W);
      }
      commit_cyc = (100000 - 1) / (H * W) + 1;
    }

    RGB BRDF_color;
		Pos pos;
    double reflection, refraction;
    double depth;
    int M;
		for (int i = 0; i < H; ++i)
			for (int j = 0; j < W; ++j) {
        const int index = i * W + j;
        CHECK_EQ(fscanf(fin, "%lf%lf%lf",
              &BRDF_color.b_, &BRDF_color.g_, &BRDF_color.r_), 3);
        BRDF_mat[0].set_float_data(index, BRDF_color.b_);
        BRDF_mat[1].set_float_data(index, BRDF_color.g_);
        BRDF_mat[2].set_float_data(index, BRDF_color.r_);
				CHECK_EQ(fscanf(fin, "%lf%lf%lf", &pos.x_, &pos.y_, &pos.z_), 3);
        CHECK_EQ(fscanf(fin, "%lf%lf%lf", &reflection, &refraction, &depth), 3);
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
        PhotonRecord tmp_photon;
        tmp_photon.reflection_ = tmp_photon.refraction_ = tmp_photon.depth_ = 0;
        tmp_photon.rgb_.b_ = tmp_photon.rgb_.g_ = tmp_photon.rgb_.r_ = 0;
        tmp_photon.pos_ = pos;
				sort(id.begin(), id.end(), Comparator(pos));
        double max_dis = 0;
        double flux = 0;
        for (int pi = 0; pi < FLAGS_photon_per_pixel; ++pi) {
          const PhotonRecord& photon = pi < id.size() ? photons[id[pi]] : tmp_photon;
          photon_dis_mat.set_float_data(index, squ_dis(pos, photon.pos_));
          max_dis = std::max(max_dis, squ_dis(pos, photon.pos_));
          flux += photon.rgb_.b_;
          photon_rrd_mat.set_float_data((pi * 3 + 0) * H * W + index, photon.reflection_);
          photon_rrd_mat.set_float_data((pi * 3 + 1) * H * W + index, photon.refraction_);
          photon_rrd_mat.set_float_data((pi * 3 + 2) * H * W + index, photon.depth_);
          photon_flux_mat[0].set_float_data(index, photon.rgb_.b_);
          photon_flux_mat[1].set_float_data(index, photon.rgb_.g_);
          photon_flux_mat[2].set_float_data(index, photon.rgb_.r_);
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
    CHECK(photon_dis_mat.SerializeToString(&out));
    for (int c = 0; c < 3; ++c) {
      txn_photon_dis->Put(filename, out);
    }
    CHECK(photon_rrd_mat.SerializeToString(&out));
    for (int c = 0; c < 3; ++c) {
      txn_photon_rrd->Put(filename, out);
    }
    for (int c = 0; c < 3; ++c) {
      CHECK(photon_flux_mat[c].SerializeToString(&out));
      txn_photon_flux->Put(filename, out);
    }
    if ((bat + 1) % commit_cyc == 0) {
      txn_conv->Commit();
      txn_BRDF->Commit();
      txn_rrd->Commit();
      txn_photon_dis->Commit();
      txn_photon_rrd->Commit();
      txn_photon_flux->Commit();
    }
  }

  if (N % commit_cyc != 0) {
    txn_conv->Commit();
    txn_BRDF->Commit();
    txn_rrd->Commit();
    txn_photon_dis->Commit();
    txn_photon_rrd->Commit();
    txn_photon_flux->Commit();
  }

  fclose(fin);

#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
