#include "caffe/layers/shuffle_channel_layer.hpp"


namespace caffe {

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size());
  CHECK(bottom.size());
  BlobProto shape;
  bottom[0]->ToProto(&shape);
  for (auto d : bottom)
    CHECK_EQ(d->ShapeEquals(shape), true);
}

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); i++)
    top[i]->ReshapeLike(*bottom[i]);
}

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<const Dtype*> bottom_data;
  vector<Dtype*> top_data;
  for (auto bottom_blob : bottom)
    bottom_data.push_back(bottom_blob->cpu_data());
  for (auto top_blob : top)
    top_data.push_back(top_blob->mutable_cpu_data());
  int dim = bottom[0]->shape(0); 
  int count = bottom[0]->count(2);
  int channels = bottom[0]->shape(1);
  for (int i = 0; i < bottom.size(); i++)
    caffe_copy(bottom[i]->count(), 
               bottom_data[i], 
               top_data[i]);
  for (int i = 0; i < dim; i++) {
    int offset = i * bottom[0]->count(1);
    for (int j = 0; j < count; j++)
      for (int k = 0; k < channels; k++) {
        int r = caffe_rng_rand() % (channels - k);
        int x = offset + k * count + j;
        int y = offset + r * count + j;
        for (auto d : top_data) {
            int t = d[x];
            d[x] = d[y];
            d[y] = t;
        }
      }
  }
}

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ShuffleChannelLayer);
#endif

INSTANTIATE_CLASS(ShuffleChannelLayer);
REGISTER_LAYER_CLASS(ShuffleChannel);

}  // namespace caffe
