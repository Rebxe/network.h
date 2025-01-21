#pragma once

#include "../defines.h"

// Sigmoid Layer

class SIGMOID
{
public:
	int siz;

public:
	inline void init(int Siz){siz=Siz;}
	inline void save(std::ofstream& ouf){writf(ouf,siz);}
	inline void load(std::ifstream& inf)
	{
		int Siz;
		readf(inf,Siz);
		init(Siz);
	}

private:
	inline void forward(int bs,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		ext_assert(siz==id*ih*iw&&siz==od*oh*ow,
			fprintf(stderr,"\
In SIGMOID::forward(...)\n\
  siz = %d\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",siz,id,ih,iw,od,oh,ow));
  		bs=std::max(bs,1);
		for(int i=0;i<bs*siz;i++) out[i]=1 / (1 + exp(-in[i]));
	}
	inline void backward(int bs,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		ext_assert(siz==id*ih*iw&&siz==od*oh*ow,
			fprintf(stderr,"\
In SIGMOID::backward(...)\n\
  siz = %d\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",siz,id,ih,iw,od,oh,ow));
		for(int i=0;i<bs*siz;i++)
		{
			float x=exp(-in[i]);
			din[i]=(1 / (1 + x)) * (1 - (1 / (1 + x)))*dout[i];
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
			&std::remove_reference<decltype(*this)>::type::forward,
			this,
			pch(1),
			pch(2),pch(3),pch(4),pch(5),
			pch(6),pch(7),pch(8),pch(9));
		res.dat->backward_f=std::bind(
			&std::remove_reference<decltype(*this)>::type::backward,
			this,
			pch(1),
			pch(2),pch(3),pch(4),pch(5),pch(6),
			pch(7),pch(8),pch(9),pch(10));
		#undef pch
		res.dat->forward();
		return res;
	}
	#ifdef ENABLE_AUTO_SL
		AUTO_SL_LAYER_CONSTRUCTER(SIGMOID)
	#endif
};
