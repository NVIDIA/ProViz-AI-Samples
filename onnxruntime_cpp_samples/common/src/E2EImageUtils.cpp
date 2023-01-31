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

#include "E2EImageUtils.h"
#include "png.h"

#include <cstring>
#include <iostream>

namespace e2eai {

	uint32_t LoadPNG(
		const char* inFilePath,
		ImageData* outData
	) {

		uint32_t img_depth;
		uint32_t img_clr_type;
  uint32_t err = LoadPNG(inFilePath, outData->width, outData->height, img_depth,
                         outData->channels, img_clr_type, &(outData->raw_data));
  if (err)
    return err;
		return 0;
	}



	uint32_t  LoadPNG(
		const char* inFilePath,
		uint32_t& outWidth,
                uint32_t &outHeight,
                uint32_t &outDepth,
                uint32_t &outChannels,
		uint32_t& outColorType,
		uint8_t** outData
	) {
		uint32_t rslt = 0;


		unsigned char header[8];
		FILE* fp = fopen(inFilePath, "rb");

		if (!fp) {
			std::cerr << "Error Reading PNG File : " << inFilePath << std::endl;
			return 1;
		}

		//Read PNG Header.
		fread(header, 1, 8, fp);
		//Check header
		if (png_sig_cmp(header, 0, 8)) {
			fclose(fp);
			std::cerr << "Error Reading PNG File : Not a png." << std::endl;
			return 1;
		}


		png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		if (!png_ptr) {
			fclose(fp);
			return 1;
		}

		png_infop info_ptr = png_create_info_struct(png_ptr);
		if (!info_ptr) {
			fclose(fp);
			return 1;
		}


		if (setjmp(png_jmpbuf(png_ptr))) {
			fclose(fp);
			return 1;
		}


		png_init_io(png_ptr, fp);
		png_set_sig_bytes(png_ptr, 8);

		png_read_info(png_ptr, info_ptr);

		outWidth = png_get_image_width(png_ptr, info_ptr);
                outChannels = png_get_channels(png_ptr, info_ptr);
		outHeight = png_get_image_height(png_ptr, info_ptr);
		outDepth = png_get_bit_depth(png_ptr, info_ptr);

		outColorType = png_get_color_type(png_ptr, info_ptr);

		uint32_t number_of_passes = png_set_interlace_handling(png_ptr);
		png_read_update_info(png_ptr, info_ptr);

		if (setjmp(png_jmpbuf(png_ptr))) {
			return 1;
		}

		uint32_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
		png_bytep* row_pointers = (png_bytep*)(malloc(sizeof(png_bytep) * outHeight));
		for (uint32_t y = 0; y < outHeight; ++y) {
			row_pointers[y] = (png_byte*)malloc(row_bytes);
		}

		uint32_t data_size = row_bytes * outHeight * sizeof(uint8_t);

		(*outData) = (uint8_t*)malloc(data_size);
		png_read_image(png_ptr, row_pointers);

		for (uint32_t y = 0; y < outHeight; ++y) {
			uint8_t* dst_ptr = (*outData) + (y * row_bytes);
                        std::memcpy(dst_ptr, row_pointers[y], row_bytes);
			free(row_pointers[y]);
		}

		free(row_pointers);

		fclose(fp);



		return rslt;
	}


	void SavePNG(const char* outPath, ImageData& inData) {

		int y;

		FILE* fp = fopen(outPath, "wb");
		if (!fp) abort();

		png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		if (!png) abort();

		png_infop info = png_create_info_struct(png);
		if (!info) abort();

		if (setjmp(png_jmpbuf(png))) abort();

		png_init_io(png, fp);

		int colorType = PNG_COLOR_TYPE_RGB;

		switch (inData.channels) {
		case 3: {
			colorType = PNG_COLOR_TYPE_RGB;
		}break;

		case 4: {
			colorType = PNG_COLOR_TYPE_RGBA;
		}break;


		case 1: {
			colorType = PNG_COLOR_TYPE_GRAY;
		}break;

		}

		// Output is 8bit depth, RGBA format.
		png_set_IHDR(
			png,
			info,
			inData.width, inData.height,
			8,
			colorType,
			PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_DEFAULT,
			PNG_FILTER_TYPE_DEFAULT
		);
		png_write_info(png, info);

		// To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
		// Use png_set_filler().
		//png_set_filler(png, 0, PNG_FILLER_AFTER);

		png_bytep* row_pointers;

		row_pointers = (png_bytep*)malloc(inData.height * sizeof(png_bytep));

		uint32_t row_bytes = inData.width * inData.channels * sizeof(uint8_t);

		uint8_t* ptr = inData.raw_data;

		for (uint32_t i = 0; i < inData.height; ++i) {
			row_pointers[i] = ptr;
			ptr += row_bytes;
		}


		png_write_image(png, row_pointers);
		png_write_end(png, NULL);


		free(row_pointers);
                inData.raw_data = nullptr;

		fclose(fp);

	}

} // namespace e2eai
