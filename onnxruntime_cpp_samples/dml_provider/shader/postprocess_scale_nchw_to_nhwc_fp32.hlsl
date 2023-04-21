/*

	Post process kernel.
	Transposes back to NHWC and re-scales input back
	to 8 bit integer range.

*/


cbuffer cbConstants {
	unsigned int slice_size;
	unsigned int channels_in;
	unsigned int channels_out;
	unsigned int pad;
};

RWBuffer<float> data_in : register(u0);
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

	data_out[DTid.x] = (unsigned int)((clamp(data_in[input_idx], -0.5f, 0.5f) + 0.5f) * 255.0f);
}
