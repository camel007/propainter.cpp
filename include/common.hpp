#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <climits>
#include <cmath>
#include <fstream>   // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "device_alternate.hpp"
#include "simple_log.hpp"

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:                                   \
    classname(const classname&);           \
    classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)     \
    char gInstantiationGuard##classname; \
    template class classname<float>;     \
    template class classname<double>

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

namespace ferrari
{

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::iterator;
using std::make_pair;
using std::ostringstream;
using std::pair;
using std::string;
using std::stringstream;
using std::vector;

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe
{
public:
    ~Caffe();

    static Caffe& Get();

    enum Brew
    {
        CPU,
        GPU
    };

    inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }

    // Returns the mode: running on CPU or GPU.
    inline static Brew mode() { return Get().mode_; }
    // The setters for the variables
    // Sets the mode. It is recommended that you don't change the mode halfway
    // into the program since that may cause allocation of pinned memory being
    // freed in a non-pinned way, which may cause problems - I haven't verified
    // it personally but better to note it here in the header file.
    inline static void set_mode(Brew mode) { Get().mode_ = mode; }
    // Sets the random seed of both boost and curand
    static void set_random_seed(const unsigned int seed);
    // Sets the device. Since we have cublas and curand stuff, set device also
    // requires us to reset those values.
    static void SetDevice(const int device_id);
    // Prints the current GPU status.
    static void DeviceQuery();
    // Check if specified device is available
    static bool CheckDevice(const int device_id);
    // Search from start_id to the highest possible device ordinal,
    // return the ordinal of the first available device.
    static int FindDevice(const int start_id = 0);

protected:
    cublasHandle_t cublas_handle_;

    Brew mode_;

private:
    // The private constructor to avoid duplicate instantiation.
    Caffe();

    DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace ferrari

#endif  // CAFFE_COMMON_HPP_