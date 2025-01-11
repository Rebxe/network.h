#pragma once

#include "../defines.h"

// Leaky_ReLU Layer

class LEAKY_RELU
{
public:
	int bs,siz;
	float a;

public:
	inline void init(int Batch_Size,int Siz,float Alpha = 0.01)
	{
		bs=Batch_Size;
		siz=Siz;
		a=Alpha;
	}
	inline void save(std::ofstream& ouf){writf(ouf,siz),writf(ouf,a);}
	inline void load(std::ifstream& inf, int Batch_Size)
	{
		int Siz;
		float Alpha;
		readf(inf,Siz),readf(inf,Alpha);
		init(Batch_Size,Siz,Alpha);
	}
	inline void forward(int Batch_Size,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		if(Batch_Size!=0) assert(Batch_Size==bs);
		assert(siz==id*ih*iw&&siz==od*oh*ow);
		for(int i=0;i<std::max(Batch_Size,1)*siz;i++) out[i]=in[i]<0?a*in[i]:in[i];
	}
	
	inline void backward(int Batch_Size,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		assert(Batch_Size==bs&&siz==id*ih*iw&&siz==od*oh*ow);
		for(int i=0;i<bs*siz;i++) din[i]=in[i]<0?a*dout[i]:dout[i];
	}
	inline val3d operator()(val3d x)
	{
		val3d res(x.d,x.h,x.w);
		res.dat->in1=x.dat;
		x.dat->oud++;
		#define pch(x) std::placeholders::_##x
		res.dat->forward_f=std::bind(
			std::remove_reference<decltype(*this)>::type::forward,
			this,
			pch(1),
			pch(2),pch(3),pch(4),pch(5),
			pch(6),pch(7),pch(8),pch(9));
		res.dat->backward_f=std::bind(
			std::remove_reference<decltype(*this)>::type::backward,
			this,
			pch(1),
			pch(2),pch(3),pch(4),pch(5),pch(6),
			pch(7),pch(8),pch(9),pch(10));
		#undef pch
		res.dat->forward();
		return res;
	}
	#ifdef ENABLE_AUTO_SL
		AUTO_SL_LAYER_CONSTRUCTER(LEAKY_RELU)
	#endif
};
