#include "config.h"
#include "model.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>
#include <fstream>

#include "cv-helpers.hpp"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API

using namespace nvinfer1;
using namespace cv;

const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
static Logger gLogger;
std::vector<cv::Mat> img_batch;

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}

void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device, float** output_buffer_host) {
  assert(engine->getNbBindings() == 2);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex = engine->getBindingIndex(kOutputTensorName);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));

  *output_buffer_host = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchSize) {
  // infer on the batch asynchronously, and DMA output back to host
  context.enqueue(batchSize, buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, std::string& img_dir, std::string& sub_type) {
  if (argc < 3) return false;
  if (std::string(argv[1]) == "-s" && argc == 5) {
    wts = std::string(argv[2]);
    engine = std::string(argv[3]);
    sub_type = std::string(argv[4]);
  } else if (std::string(argv[1]) == "-d" && argc == 3) {
    engine = std::string(argv[2]);
    // img_dir = std::string(argv[3]);
  } else {
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  cudaSetDevice(kGpuId);

  std::string wts_name = "";
  std::string engine_name = "";
  std::string img_dir;
  std::string sub_type = "";

  if (!parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type)) {
    std::cerr << "Arguments not right!" << std::endl;
    std::cerr << "./yolov7 -s [.wts] [.engine] [t/v7/x/w6/e6/d6/e6e]  // serialize model to plan file" << std::endl;
    std::cerr << "./yolov7 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
    return -1;
  }

// Declare depth colorizer for pretty visualization of depth data
rs2::colorizer color_map;

// Declare RealSense pipeline, encapsulating the actual device and sensors
rs2::pipeline pipe;
// Start streaming with default recommended configuration
pipe.start();

const auto window_name = "Realsense";
namedWindow(window_name, WINDOW_AUTOSIZE);

  // Deserialize the engine from file
  IRuntime* runtime = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  deserialize_engine(engine_name, &runtime, &engine, &context);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cuda_preprocess_init(kMaxInputImageSize);

  // Prepare cpu and gpu buffers
  float* device_buffers[2];
  float* output_buffer_host = nullptr;
  prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host);

    while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        // Getting frame from realsense
        rs2::frameset data = pipe.wait_for_frames(); 
        rs2::frame color = data.get_color_frame();
        auto color_mat = frame_to_mat(color);

        // img_batch.push_back(color_mat);

        // Preprocess
        // cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);

        cuda_preprocess(color_mat.ptr(), color_mat.cols, color_mat.rows, device_buffers[0], kInputW, kInputH, stream);

        // Run inference
        auto start = std::chrono::system_clock::now();
        infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // NMS
        std::vector<std::vector<Detection>> res_batch;
        batch_nms(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh);

        // Draw bounding boxes
        draw_bbox_test(color_mat, res_batch);

        // Update the window with new data
        imshow(window_name, color_mat);
    }

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CUDA_CHECK(cudaFree(device_buffers[0]));
  CUDA_CHECK(cudaFree(device_buffers[1]));
  delete[] output_buffer_host;
  cuda_preprocess_destroy();
  // Destroy the engine
  delete context;
  delete engine;
  delete runtime;

  // Print histogram of the output distribution
  //std::cout << "\nOutput:\n\n";
  //for (unsigned int i = 0; i < kOutputSize; i++)
  //{
  //    std::cout << prob[i] << ", ";
  //    if (i % 10 == 0) std::cout << std::endl;
  //}
  //std::cout << std::endl;

  return 0;
}

