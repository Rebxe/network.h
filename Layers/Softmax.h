#pragma once

#include "../defines.h"

// Softmax Layer

class SOFTMAX
{
public:
	int d,h,w;

public:
	inline void init(SHAPE3D Input){d=std::get<0>(Input),h=std::get<1>(Input),w=std::get<2>(Input);}
	inline void save(std::ofstream& ouf){writf(ouf,(SHAPE3D){d,h,w});}
	inline void load(std::ifstream& inf)
	{
		SHAPE3D Input;
		readf(inf,Input);
		init(Input);
	}

private:
	inline void forward(int bs,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		ext_assert(d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow,
			fprintf(stderr,"\
In Softmax::forward(...)\n\
  shape = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",d,h,w,id,ih,iw,od,oh,ow));
  		bs=std::max(bs,1);
		for(int tb=0;tb<bs;tb++)
		{
			int adb=tb*d*h*w;
			for(int i=0;i<h;i++)
			{
				for(int j=0;j<w;j++)
				{
					int ad=adb+i*w*j;
					float sm=0;
					for(int k=0;k<d;k++) sm+=exp(in[ad+k*h*w]);
					for(int k=0;k<d;k++) out[ad+k*h*w]=exp(in[ad+k*h*w])/sm;
				}
			}
		}
	}
	inline void backward(int bs,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		ext_assert(d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow,
			fprintf(stderr,"\
In Softmax::backward(...)\n\
  shape = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",d,h,w,id,ih,iw,od,oh,ow));
		for(int tb=0;tb<bs;tb++)
		{
			int adb=tb*d*h*w;
			for(int i=0;i<h;i++)
			{
				for(int j=0;j<w;j++)
				{
					int ad=adb+i*w*j;
					float sm=0;
					for(int k=0;k<d;k++) sm+=exp(in[ad+k*h*w]);
					float smd=0;
					for(int k=0;k<d;k++) smd+=-exp(in[ad+k*h*w])/sm*dout[ad+k*h*w];
					for(int k=0;k<d;k++) din[ad+k*h*w]=(dout[ad+k*h*w]+smd)*exp(in[ad+k*h*w])/sm;
				}
			}
		}
	}

public:
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
		AUTO_SL_LAYER_CONSTRUCTER(SOFTMAX)
	#endif
};
