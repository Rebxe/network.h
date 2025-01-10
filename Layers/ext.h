#pragma once

#include "../defines.h"

// Extend Layer

class EXT
{
public:
	int bs, ind, inh, inw;
	int filx, fily;
	int ouh, ouw;

public:
	inline void init(int Batch_Size,SHAPE3D Input,std::pair<int,int> Fill)
	{
		bs = Batch_Size;
		ind = std::get<0>(Input), inh = std::get<1>(Input), inw = std::get<2>(Input);
		filx = Fill.first, fily = Fill.second;
		ouh = inh * filx, ouw = inw * fily;
	}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,(SHAPE3D){ind,inh,inw});
		writf(ouf,std::make_pair(filx,fily));
	}
	inline void load(std::ifstream& inf, int Batch_Size)
	{
		SHAPE3D Input;
		std::pair<int,int> Fill;
		readf(inf,Input),readf(inf,Fill);
		init(Batch_Size,Input,Fill);
	}
	inline void forward(int Batch_Size,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out,
						bool test)
	{
		assert(Batch_Size==bs&&ind==id&&inh==ih&&inw==iw&&ind==od&&ouh==oh&&ouw==ow);
		for(int tb=0;tb<bs;tb++)
		{
			int adi=tb*ind*inh*inw,ado=tb*ind*ouh*ouw;
			for (int d = 0; d < ind; d++) for (int i = 0; i < inh; i++) for (int j = 0; j < inw; j++)
			{
				float& pre = in[adi + d * inh * inw + i * inw + j];
				int ad = d * ouh * ouw;
				for (int k = 0; k < filx; k++) for (int l = 0; l < fily; l++) out[ado + ad + (i * filx + k) * ouw + j * fily + l] = pre;
			}
			if(test) break; 
		}
	}
	inline void backward(int Batch_Size,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		assert(Batch_Size==bs&&ind==id&&inh==ih&&inw==iw&&ind==od&&ouh==oh&&ouw==ow);
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
			pch(6),pch(7),pch(8),pch(9),
			pch(10));
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
