#pragma once

#include "../defines.h"

// Bias Layer

class BIAS
{
public:
	int d,h,w;
	float *b;

private:
	float *tmpb;
	inline void initmem(float *&wei, float *&tmp)
	{
		b=wei,tmpb=tmp;
		wei+=d,tmp+=d;
	}

public:
	inline void init(int &m,SHAPE3D Input)
	{
		d=std::get<0>(Input),h=std::get<1>(Input),w=std::get<2>(Input);
		m+=d;
	}
	inline void build(float *&wei, float *&tmp)
	{
		initmem(wei,tmp);
		// init w
		memset(b,0,sizeof(float)*d);
	}
	inline void save(std::ofstream& ouf){writf(ouf,SHAPE3D(d,h,w));}
	inline void load(std::ifstream& inf,float *&wei,float *&tmp)
	{
		SHAPE3D Input;
		readf(inf,Input);
		int nou=0;
		init(nou,Input);
		initmem(wei,tmp);
	}

private:
	inline void forward(int bs,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		ext_assert(d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow,
			fprintf(stderr,"\
In BIAS::forward(...)\n\
  shape = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",d,h,w,id,ih,iw,od,oh,ow));
		bs=std::max(bs,1);
		for(int t=0;t<bs;t++)
		{
			int adt=t*d*h*w;
			for(int i=0;i<d;i++)
			{
				int ad=i*h*w;
				for(int j=0;j<h*w;j++) out[adt+ad+j]=in[adt+ad+j]+b[i];
			}
		}
	}
	inline void backward(int bs,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		ext_assert(d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow,
			fprintf(stderr,"\
In BIAS::backward(...)\n\
  shape = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",d,h,w,id,ih,iw,od,oh,ow));
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
		AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(BIAS)
	#endif
};
