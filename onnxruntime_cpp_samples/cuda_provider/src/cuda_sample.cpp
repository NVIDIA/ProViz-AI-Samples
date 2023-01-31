/*###############################################################################
#
# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################*/

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <iostream>

#include "CUDAInference.h"
#include "E2EImageUtils.h"
#include "FormatConversion.cuh"
#include "TRTInference.h"
#include "cuda_helpers.cuh"
#include "onnx_helpers.h"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

static argparse::ArgumentParser parse_args(int argc, const char *const argv[]) {
  argparse::ArgumentParser program("cuda_sample");
  program.add_argument("--images")
      .help("Path to either a png file or a folfer with png's to be inferred.")
      .default_value(std::string("./images"));
  program.add_argument("--image-out")
      .help("Output folder to put inferred result images.")
      .default_value(std::string("./"));
  program.add_argument("--gpu-id")
      .help("ID of the GPU to use for inference.")
      .default_value(static_cast<uint32_t>(0));
  program.add_argument("--profile")
      .help("Set this flag to print all timings.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--fp16")
      .help("Set this flag to print all timings.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--trt")
      .help("Set this flag to print all timings.")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--scale")
      .help("Scale factor that the ONNX model applies.")
      .default_value(2)
      .scan<'i', int>();
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }
  return program;
}

int main(int argc, const char *const argv[]) {
  auto args = parse_args(argc, argv);

  const auto image_path = fs::path(args.get("--images"));
  const auto out_folder = fs::path(args.get("--image-out"));
  auto gpu_id = args.get<uint32_t>("--gpu-id");
  auto scale = args.get<int>("--scale");
  auto fp16 = args.get<bool>("--fp16");
  auto trt = args.get<bool>("--trt");

  /// Loading different model if fp16 is used
  const std::string model_path =
      fp16 ? "./models/ISRModel2X_fp16.onnx" : "./models/ISRModel2X.onnx";

  std::vector<fs::path> image_files;
  std::string image_path_extension = image_path.extension().string();

  if (fs::is_directory(image_path)) {
    for (auto const &dir_entry :
         std::filesystem::directory_iterator{image_path}) {
      auto cur_path = dir_entry.path();
      std::string extension = cur_path.extension().string();
      if (std::strcmp(extension.c_str(), ".png") == 0) {
        image_files.push_back(cur_path);
      }
    }
  } else if (fs::exists(image_path) &&
             std::strcmp(image_path_extension.c_str(), ".png") == 0) {
    image_files.push_back(image_path);
  }
  if (image_files.empty()) {
    std::cerr << "given path is neither a folder nor an image (.png)"
              << std::endl;
    return -1;
  } else {
    std::cout << "Images to process:" << std::endl;
    for (auto &f : image_files)
      std::cout << "\t" << f << std::endl;
  }

  /// Load images - assuming all images are the same size !
  std::vector<e2eai::ImageData> img_data{image_files.size()};
  for (int idx = 0; idx < image_files.size(); ++idx) {
    std::string cur_img_path = image_files[idx].string();
    e2eai::LoadPNG(cur_img_path.c_str(), &img_data[idx]);
  }
  size_t image_bytes = img_data[0].sizeInBytes();
  /// input shape
  std::vector<int64_t> im_shape = img_data[0].shape();
  size_t image_bytes_out = image_bytes * scale * scale;
  /// output shape is dependent on the scale factor as we are using a super res
  /// network
  std::vector<int64_t> sr_shape = im_shape;
  sr_shape[2] *= scale;
  sr_shape[3] *= scale;
  /// Allocate pinned memory on the CPU for output tensors to make PCI copies as
  /// performant as possible
  std::vector<e2eai::ImageData> img_data_out;
  for (int idx = 0; idx < image_files.size(); ++idx) {
    e2eai::ImageData temp_img;
    temp_img.channels = sr_shape[1];
    temp_img.height = sr_shape[2];
    temp_img.width = sr_shape[3];
    cudaMallocHost(&temp_img.raw_data, temp_img.sizeInBytes());
    img_data_out.push_back(std::move(temp_img));
  }

  /// Running ORT Inference and pre-/postprocessing steps
  /// In our case this is only transforming the input from NCHW to NHWC and
  /// back.
  std::unique_ptr<ProcessingTemplate> preprocess;
  std::unique_ptr<ProcessingTemplate> postprocess;
  if (fp16) {
    using inference_dtype = half;
    preprocess = std::make_unique<NCHW_to_NHWC<uint8_t, inference_dtype>>();
    postprocess = std::make_unique<NHWC_to_NCHW<inference_dtype, uint8_t>>();
  } else {
    using inference_dtype = float;
    preprocess = std::make_unique<NCHW_to_NHWC<uint8_t, inference_dtype>>();
    postprocess = std::make_unique<NHWC_to_NCHW<inference_dtype, uint8_t>>();
  }

  /// Instantiate the inference engine with TensorRT EP or CUDA EP depending on
  /// the command line
  std::unique_ptr<NVInference> inference_module;
  try {
    if (trt) {
      inference_module = std::make_unique<TRTInference>(
          model_path, im_shape, sr_shape, preprocess.get(), postprocess.get(),
          gpu_id, fp16);
    } else {
      bool use_exhaustive_kernel_search = false;
      inference_module = std::make_unique<CUDAInference>(
          model_path, im_shape, sr_shape, preprocess.get(), postprocess.get(),
          gpu_id, use_exhaustive_kernel_search);
    }

    /// Allocating GPU resident output tensor
    /// (uint8 as this is after postprocessing)
    inference_module->allocateOutputTensor(sr_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    cudaStream_t compute_stream = inference_module->fetchComputeStream();

    /// Streams to do PCI copies
    cudaStream_t upload_stream, download_stream;
    CheckCUDA(cudaStreamCreate(&upload_stream));
    CheckCUDA(cudaStreamCreate(&download_stream));

    /// Event to signal finished PCI copies and finished inference
    cudaEvent_t inference_complete_event;
    CheckCUDA(cudaEventCreate(&inference_complete_event));
    cudaEvent_t upload_complete_event;
    CheckCUDA(cudaEventCreate(&upload_complete_event));
    cudaEvent_t download_complete_event;
    CheckCUDA(cudaEventCreate(&download_complete_event));
    CheckCUDA(cudaEventRecord(download_complete_event));

    /// GPU resident input tensor
    inference_module->allocateInputTensor(im_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    Ort::Value *img_tensor = inference_module->fetchInputTensorAsync();
    auto *raw_uint = img_tensor->GetTensorMutableData<uint8_t>();

    /// Upload input and trigger an event afterwards (async)
    CheckCUDA(cudaMemcpyAsync(raw_uint, img_data[0].raw_data, image_bytes,
                              cudaMemcpyHostToDevice, upload_stream));
    CheckCUDA(cudaEventRecord(upload_complete_event, upload_stream));
    for (int idx = 0; idx < img_data.size();) {
      Ort::RunOptions run_options;
      /// Wait for input event to be trigger (async)
      CheckCUDA(cudaStreamWaitEvent(compute_stream, upload_complete_event));

      /// Set the event that has to be checked before overwriting output buffer
      /// (this event will be checked in inference_module->runAsync)
      inference_module->setDownloadedEvent(&download_complete_event);

      /// Run inference but record event as soon as input buffer has been
      /// consumed and can be overwritten (async)
      inference_module->runAsync(img_tensor, &run_options);

      /// Record event to capture a finished inference (async)
      CheckCUDA(cudaEventRecord(inference_complete_event, compute_stream));

      /// When profiling no async execution is possible !
      /// getTimings synchronizes
      if (args.get<bool>("--profile")) {
        auto timings = inference_module->getTimings();
        std::cout << "Measured processing steps for image " << idx << std::endl;
        for (const auto &[k, v] : timings) {
          std::cout << "\t" << k << ":\t" << v << "ms" << std::endl;
        }
      }
      /// Leave loop if last inference was submitted
      if (++idx >= img_data.size())
        break;

      /// Wait until the input buffer has been consumed and upload next buffer (async)
      auto consumed_ev = inference_module->getConsumedEvent();
      CheckCUDA(cudaStreamWaitEvent(upload_stream, consumed_ev));
      CheckCUDA(cudaMemcpyAsync(raw_uint, img_data[idx].raw_data, image_bytes,
                                cudaMemcpyHostToDevice, upload_stream));
      /// Capture event when upload is finished to check in next iteration (async)
      CheckCUDA(cudaEventRecord(upload_complete_event, upload_stream));

      /// GPU resident output tensor
      Ort::Value *out_val = inference_module->fetchOutputTensorAsync();
      const auto *raw_out = out_val->GetTensorData<uint8_t>();

      /// Download result as soon as inference is completed (async)
      CheckCUDA(cudaStreamWaitEvent(download_stream, inference_complete_event));
      CheckCUDA(cudaMemcpyAsync(img_data_out[idx - 1].raw_data, raw_out,
                                image_bytes_out, cudaMemcpyDeviceToHost,
                                download_stream));

      /// Record event as soon as output buffer has been read and can be reused (async)
      CheckCUDA(cudaEventRecord(download_complete_event, download_stream));
    }
    /// GPU resident output tensor
    Ort::Value *out_val = inference_module->fetchOutputTensorAsync();
    const auto *raw_out = out_val->GetTensorData<uint8_t>();

    /// Synchronize everything to stop pipeline and wait for all inferences to finish (sync)
    CheckCUDA(cudaEventSynchronize(inference_complete_event));
    CheckCUDA(cudaMemcpyAsync(img_data_out[img_data.size() - 1].raw_data, raw_out,
                              image_bytes_out, cudaMemcpyDeviceToHost,
                              download_stream));
    CheckCUDA(cudaStreamSynchronize(download_stream));

    /// Write out images to disc
    for (int idx = 0; idx < img_data.size(); ++idx) {
      fs::path file_name(fmt::format("image_{:02d}.png", idx));
      fs::path full_path = out_folder / file_name;
      std::string full_path_str = full_path.string();
      e2eai::SavePNG(full_path_str.c_str(), img_data_out[idx]);
    }
  } catch (std::exception &e) {
  std::cerr << e.what();
  return -1;
  }
  return 0;
}
