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

#ifndef __E2EIMAGEUTILS_H__
#define __E2EIMAGEUTILS_H__

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace e2eai {

struct ImageData {
  uint8_t* raw_data;
  uint32_t width;
  uint32_t height;
  uint32_t channels;
  std::string name;

  ImageData() = default;

  ImageData(std::vector<int64_t> shape) {
    width = shape[3];
    height = shape[2];
    channels = shape[1];
    raw_data = (uint8_t *)malloc(height * width * channels);
  }

  ImageData(ImageData &rhs) = delete;

  ImageData(ImageData &&rhs) {
    this->channels = rhs.channels;
    this->height = rhs.height;
    this->width = rhs.width;
    this->raw_data = rhs.raw_data;
    rhs.raw_data = nullptr;
    this->name = rhs.name;
    rhs.name = "";
  };

  ImageData(uint32_t in_width, uint32_t in_height, uint32_t in_channels) {
    width = in_width;
    height = in_height;
    channels = in_channels;
    raw_data = (uint8_t *)malloc(height * width * channels);
  }

  ~ImageData() {
    if (raw_data != nullptr)
      free(raw_data);
  }

  [[nodiscard]] inline auto
  sizeInBytes() const {
    return sizeof(*raw_data) * width * height * channels;
  }

  [[nodiscard]] inline auto shape() const -> std::vector<int64_t> {
    return std::vector<int64_t>{1, channels, height, width};
  }
};

uint32_t LoadPNG(
    const char* inFilePath,
    ImageData* outData);

uint32_t LoadPNG(
    const char* inFilePath,
    uint32_t& outWidth,
    uint32_t& outHeight,
    uint32_t& outDepth,
    uint32_t& outChannels,
    uint32_t& outColorType,
    uint8_t** outData);

void SavePNG(
    const char* outFilePath,
    ImageData& inData);

}  // namespace e2eai

#endif
