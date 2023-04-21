/*

	Pre process kernel.
	Transposes back to NCHW and scales input 
	to -0.5<->0.5 floating point range.

*/


cbuffer cbConstants {
	unsigned int slice_size;
	unsigned int channels_in;
	unsigned int channels_out;
	unsigned int pad;
};


RWBuffer<uint> data_in : register(u0);
RWBuffer<float16_t> data_out : register(u1);


[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 GrpID : SV_GROUPID)
{

	unsigned int input_count = slice_size * channels_in;
	unsigned int total_count = slice_size * channels_out;

	if (DTid.x >= total_count) return;

	int input_channel = DTid.x % channels_in;

	unsigned int output_idx = DTid.x; 
	float16_t output_value = 0.0h;

	if (DTid.x < input_count) {
		output_idx = (DTid.x / channels_in) + (input_channel * slice_size);
		output_value = (half(data_in.Load(DTid.x))/255.0h)-0.5h;
	}

	data_out[output_idx] = (output_value);
}
