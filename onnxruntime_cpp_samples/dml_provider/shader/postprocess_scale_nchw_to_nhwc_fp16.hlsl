/*
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

RWBuffer<float16_t> data_in : register(u0);
RWBuffer<uint> data_out : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 GrpID : SV_GROUPID)
{

	
	//Channels out wants to be 3 for both input and output.
	unsigned int input_count = slice_size * channels_out; 
	unsigned int total_count = slice_size * channels_out;

	if (DTid.x >= total_count) return;

	int input_channel = DTid.x % channels_out;
	unsigned int input_idx = (DTid.x / channels_in) + (input_channel * slice_size);

	data_out[DTid.x] = (unsigned int)((clamp(data_in[input_idx], -0.5h, 0.5h) + 0.5h) * 255.0h);
}
