/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


cbuffer cbConstants {
	unsigned int slice_size;
	unsigned int channels_in;
	unsigned int channels_out;
	unsigned int pad;
};


RWBuffer<uint> data_in : register(u0);
RWBuffer<float> data_out : register(u1);


[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 GrpID : SV_GROUPID)
{

	unsigned int input_count = slice_size * channels_in;
	unsigned int total_count = slice_size * channels_out;

	if (DTid.x >= total_count) return;

	int input_channel = DTid.x % channels_in;

	unsigned int output_idx = DTid.x; 
	float output_value = 0.0f;

	if (DTid.x < input_count) {
		output_idx = (DTid.x / channels_in) + (input_channel * slice_size);
		output_value = ((data_in.Load(DTid.x))/255.0f)-0.5f;
	}

	data_out[output_idx] = (output_value);
}
