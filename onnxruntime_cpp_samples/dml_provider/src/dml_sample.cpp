/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "E2EImageUtils.h"
#include "OrtInference.h"
#include <argparse/argparse.hpp>
#include <filesystem>

using namespace e2eai;
namespace fs = std::filesystem;

static argparse::ArgumentParser parse_args(int argc, const char *const argv[]) {
  argparse::ArgumentParser program("cuda_sample");
  program.add_argument("--image")
      .help("Path to either a png file or a folfer with png's to be inferred.")
      .default_value(std::string("./images/noisey_20percent.png"));
  program.add_argument("--image-out")
      .help("Output folder to put inferred result images.")
      .default_value(std::string("./"));
  program.add_argument("--fp16")
      .help("Set this flag to print all timings.")
      .default_value(false)
      .implicit_value(true);
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

  const auto image_path = fs::path(args.get("--image")).string();
  const auto out_folder = fs::path(args.get("--image-out"));
  auto fp16 = args.get<bool>("--fp16");

  OrtInference ort_inf;

  fs::path output_image_path =
      out_folder / "denoised_and_superresed_ort_dml_fp32.png";
  EnginePath ort_model_path = "./models/HierTest4EncNoBNModel.onnx";
  EnginePath ort_model_path2 = "./models/ISRModel2X.onnx";
  if (fp16) {
    output_image_path = out_folder / "denoised_and_superresed_ort_dml_fp16.png";
    ort_model_path = "./models/HierTest4EncNoBNModel_fp16.onnx";
    ort_model_path2 = "./models/ISRModel2X_fp16.onnx";
  }
  try {
    ort_inf.LoadInput(image_path.c_str());
    ort_inf.InitInference(ort_model_path, ort_model_path2);

    ort_inf.InitPreprocessCommands();
    ort_inf.InitPostProcessCommands();

    ort_inf.InitBindings();

    ort_inf.Run(nullptr);

    ort_inf.SaveResult(output_image_path.string().c_str());
  } catch (std::runtime_error &e) {
    std::cerr << e.what() << std::endl;
  };
  return 0;
}
