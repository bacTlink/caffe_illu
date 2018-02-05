#include <algorithm>
#include <vector>

#include "caffe/layers/photon_mapping_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PhotonMappingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PhotonMappingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> new_shape = bottom[0]->shape();
  CHECK_EQ(new_shape.size(), 4);
  new_shape[1] = 1;
  top[0]->Reshape(new_shape);
}

/*
 * bottom:
 *   0: photon squ dis data
 *   1: photon flux data
 */
template <typename Dtype>
void PhotonMappingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* squ_dis_data = bottom[0]->cpu_data();
  const Dtype* flux_data = bottom[1]->cpu_data();
  CHECK_EQ(bottom[0]->num_axes(), 4);
  CHECK_EQ(bottom[1]->num_axes(), 4);
  for (int i = 0; i < 4; ++i)
    CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i));
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->shape(0);
  const int photon_num = bottom[0]->shape(1) / 5;
  const int H = bottom[0]->shape(2);
  const int W = bottom[0]->shape(3);

  for (int n = 0; n < num; ++n) {
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j) {
        double max_squ_dis = 0;
        double flux = 0;
        const int index = i * W + j;
        for (int p = 0; p < photon_num; ++p) {
          max_squ_dis = std::max(max_squ_dis,
              static_cast<double>(squ_dis_data[index]));
          flux += static_cast<double>(flux_data[index]);
        }
        top_data[index] = max_squ_dis > 1e-6 ? flux / max_squ_dis : 0;
      }
  }
}

INSTANTIATE_CLASS(PhotonMappingLayer);
REGISTER_LAYER_CLASS(PhotonMapping);
}  // namespace caffe
