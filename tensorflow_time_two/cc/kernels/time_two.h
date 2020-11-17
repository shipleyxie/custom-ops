// kernel_example.h
#ifndef KERNEL_TIME_TWO_H_
#define KERNEL_TIME_TWO_H_

namespace tensorflow {

namespace functor {

    template <typename Device, typename T> struct ZeroOutFunctor {
        void operator()(const Device& d, int batch_size, int data_len,
            const T* input_tensor, const T* weights_tensor, T* output_tensor);
    };
} // namespace functor

} // namespace tensorflow

#endif // KERNEL_TIME_TWO_H_
