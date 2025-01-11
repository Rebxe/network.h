#pragma once

#include "../defines.h"

// Bias Layer

class BIAS
{
public:
	int bs,d,h,w;
	float *b;

private:
	float *tmpb;
	inline void initmem(float *&wei, float *&tmp)
	{
		b=wei,tmpb=tmp;
		wei+=d,tmp+=d;
	}

public:
	inline void init(int &m,int Batch_Size,SHAPE3D Input)
	{
		bs=Batch_Size;
		d=std::get<0>(Input),h=std::get<1>(Input),w=std::get<2>(Input);
		m+=d;
	}
	inline void build(float *&wei, float *&tmp)
	{
		initmem(wei,tmp);
		// init w
		memset(b,0,sizeof(float)*d);
	}
	inline void save(std::ofstream& ouf){writf(ouf,(SHAPE3D){d,h,w});}
	inline void load(std::ifstream& inf, int Batch_Size, float *&wei, float *&tmp)
	{
		SHAPE3D Input;
		readf(inf,Input);
		int nou=0;
		init(nou,Batch_Size,Input);
		initmem(wei, tmp);
	}
	inline void forward(int Batch_Size,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		if(Batch_Size!=0) assert(Batch_Size==bs);
		assert(d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow);
		for(int t=0;t<std::max(Batch_Size,1);t++)
		{
			int adt=t*d*h*w;
			for(int i=0;i<d;i++)
			{
				int ad=i*h*w;
				for(int j=0;j<h*w;j++) out[adt+ad+j]=in[adt+ad+j]+b[i];
			}
		}
	}
	inline void backward(int Batch_Size,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		assert(Batch_Size==bs&&d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow);
		for(int t=0;t<bs;t++)
		{
			int adt=t*d*h*w;
			for(int i=0;i<d;i++)
			{
				int ad=i*h*w;
				for(int j=0;j<h*w;j++) din[adt+ad+j]=dout[adt+ad+j],tmpb[i]+=dout[adt+ad+j];
			}
		}
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
		AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(BIAS)
	#endif
};
