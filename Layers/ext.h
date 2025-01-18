#pragma once

#include "../defines.h"

// Extend Layer

class EXT
{
public:
	int ind, inh, inw;
	int filx, fily;
	int ouh, ouw;

public:
	inline void init(SHAPE3D Input,std::pair<int,int> Fill)
	{
		ind = std::get<0>(Input), inh = std::get<1>(Input), inw = std::get<2>(Input);
		filx = Fill.first, fily = Fill.second;
		ouh = inh * filx, ouw = inw * fily;
	}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,(SHAPE3D){ind,inh,inw});
		writf(ouf,std::make_pair(filx,fily));
	}
	inline void load(std::ifstream& inf)
	{
		SHAPE3D Input;
		std::pair<int,int> Fill;
		readf(inf,Input),readf(inf,Fill);
		init(Input,Fill);
	}

private:
	inline void forward(int bs,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		assert(ind==id&&inh==ih&&inw==iw&&ind==od&&ouh==oh&&ouw==ow);
		ext_assert(ind==id&&inh==ih&&inw==iw&&ind==od&&ouh==oh&&ouw==ow,
			fprintf(stderr,"\
In EXT::forward(...)\n\
  in  = [%d * %d * %d]\n\
  out = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",ind,inh,inw,ind,ouh,ouw,id,ih,iw,od,oh,ow));
		bs=std::max(bs,1);
		for(int tb=0;tb<bs;tb++)
		{
			int adi=tb*ind*inh*inw,ado=tb*ind*ouh*ouw;
			for (int d = 0; d < ind; d++) for (int i = 0; i < inh; i++) for (int j = 0; j < inw; j++)
			{
				float& pre = in[adi + d * inh * inw + i * inw + j];
				int ad = d * ouh * ouw;
				for (int k = 0; k < filx; k++) for (int l = 0; l < fily; l++) out[ado + ad + (i * filx + k) * ouw + j * fily + l] = pre;
			}
		}
	}
	inline void backward(int bs,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		assert(ind==id&&inh==ih&&inw==iw&&ind==od&&ouh==oh&&ouw==ow);
		ext_assert(ind==id&&inh==ih&&inw==iw&&ind==od&&ouh==oh&&ouw==ow,
			fprintf(stderr,"\
In EXT::backward(...)\n\
  in  = [%d * %d * %d]\n\
  out = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",ind,inh,inw,ind,ouh,ouw,id,ih,iw,od,oh,ow));
		for(int tb=0;tb<bs;tb++)
		{
			int adi=tb*ind*inh*inw,ado=tb*ind*ouh*ouw;
			for (int d = 0; d < ind; d++) for (int i = 0; i < inh; i++) for (int j = 0; j < inw; j++)
			{
				float& pre = din[adi + d * inh * inw + i * inw + j];
				pre=0;
				int ad = d * ouh * ouw;
				for (int k = 0; k < filx; k++) for (int l = 0; l < fily; l++) pre += dout[ado + ad + (i * filx + k) * ouw + j * fily + l];
			}
		}
	}

public:
	inline val3d operator()(val3d x)
	{
		val3d res(ind,ouh,ouw);
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
		AUTO_SL_LAYER_CONSTRUCTER(EXT)
	#endif
};
