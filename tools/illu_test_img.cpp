#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV
#include <vector>
#include <sstream>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/util/db.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

DEFINE_string(model_file, "net.prototxt", "Model file");
DEFINE_string(trained_file, "converge.caffemodel", "Trained file");
DEFINE_string(data_dir, "data/illu/", "Data directory");
DEFINE_string(data_prefix, "raw_data", "Data file prefix");
DEFINE_string(dst_dir, "data/illu/", "Output image directory");

#ifdef USE_OPENCV

class Projector {
 public:
  Projector(const string& model_file,
            const string& trained_file);

  void Project(const string& lmdb_photon_dis,
               const string& lmdb_photon_flux,
               const string& lmdb_photon_rrd);

 private:
  void SetCursor(const int& num,
                 const string& file);

 private:
  int count_;
  shared_ptr<Net<float> > net_;
  shared_ptr<db::DB> dbs_[3];
  shared_ptr<db::Cursor> cursors_[3];
  Datum datums_[3];
};

Projector::Projector(const string& model_file,
                     const string& trained_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 3) << "Network should have exactly four input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  count_ = 0;
}

void Projector::SetCursor(const int& num,
                          const string& file) {
  dbs_[num].reset(db::GetDB("lmdb"));
  dbs_[num]->Open(file, db::READ);
  cursors_[num].reset(dbs_[num]->NewCursor());
}

void Projector::Project(const string& lmdb_photon_dis,
                        const string& lmdb_photon_flux,
                        const string& lmdb_photon_rrd) {
  SetCursor(0, lmdb_photon_dis);
  SetCursor(1, lmdb_photon_flux);
  SetCursor(2, lmdb_photon_rrd);

  while (cursors_[0]->valid()) {
    for (int i = 0; i < 3; ++i)
      CHECK(cursors_[i]->valid());

    // Put value into input layers
    for (int i = 0; i < 3; ++i) {
      Blob<float>* input_layer = net_->input_blobs()[i];
      datums_[i].ParseFromString(cursors_[i]->value());
      input_layer->Reshape(1,
                           datums_[i].channels(),
                           datums_[i].height(),
                           datums_[i].width());
    }
    net_->Reshape();

    int H, W;
    H = datums_[0].height();
    W = datums_[0].height();
    for (int i = 0; i < 3; ++i) {
      CHECK_EQ(H, datums_[i].height());
      CHECK_EQ(W, datums_[i].width());
    }

    for (int i = 0; i < 3; ++i) {
      Blob<float>* input_layer = net_->input_blobs()[i];
      float* input_data = input_layer->mutable_cpu_data();
      const float* datum_data = datums_[i].float_data().data();
      const int size = datums_[i].float_data_size();
      memcpy(input_data, datum_data, sizeof(float) * size);
    }
    net_->Forward();

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(output_layer->shape(1), 1);
    CHECK_EQ(output_layer->shape(2), H);
    CHECK_EQ(output_layer->shape(3), W);
    const float* output_data = output_layer->cpu_data();
    vector<float> vec(output_data, output_data + H * W);
    cv::Mat img(H, W, CV_32FC1, &vec.front());
    std::stringstream filename;
    filename << FLAGS_dst_dir << FLAGS_data_prefix << "_test_" << count_ << ".bmp";
    imwrite(filename.str(), img);

    ++count_;
    for (int i = 0; i < 3; ++i)
      cursors_[i]->Next();
  }
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Show test image\n"
        "Usage:\n"
        "    illu_test_img [FLAGS]\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 1) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/illu_test_img");
    return 1;
  }

  const string model_file   = FLAGS_model_file;
  const string trained_file = FLAGS_trained_file;

  const string photon_dis_file = FLAGS_data_dir + FLAGS_data_prefix + "_photon_dis";
  const string photon_flux_file = FLAGS_data_dir + FLAGS_data_prefix + "_photon_flux";
  const string photon_rrd_file = FLAGS_data_dir + FLAGS_data_prefix + "_photon_rrd";

  Projector projector(model_file, trained_file);
  projector.Project(photon_dis_file, photon_flux_file, photon_rrd_file);

  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
