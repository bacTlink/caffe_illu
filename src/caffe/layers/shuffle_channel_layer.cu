#include "caffe/layers/shuffle_channel_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void kernel_shuffle_channel(const int num,
        int channels) {};

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {
    //TODO 
    this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void ShuffleChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
        const vector<bool>& propagate_down, 
        const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ShuffleChannelLayer);
}  // namespace caffe
