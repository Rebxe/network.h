#pragma once

#include "../defines.h"

// Full Connection layer

class FC
{
public:
	int bs, ins, ous;
	bool ub;
	float * w;

private:
	float * tmpw;
	inline void initmem(float *&wei,float *&tmp)
	{
		w=wei,tmpw=tmp;
		wei+=ins*ous,tmp+=ins*ous;
	}

public:
	inline void init(int &m, int Batch_Size, int INS, int OUS)
	{
		bs = Batch_Size;
		ins = INS, ous = OUS;
		m+=ins*ous;
	}
	inline void build(float *&wei,float *&tmp,int InitType = INIT_HE)
	{
		initmem(wei,tmp);
		// init w
		std::default_random_engine gen;
		std::normal_distribution<float> wer;
		if (InitType == INIT_HE) wer = std::normal_distribution<float>(0, sqrt(2 / (float)ins));
		else                     wer = std::normal_distribution<float>(0, sqrt(1 / (float)ins));
		gen.seed(time(NULL));
		for (int i = 0; i < ins * ous; i++) w[i] = wer(gen);
	}
	inline void save(std::ofstream& ouf){writf(ouf,ins),writf(ouf,ous);}
	inline void load(std::ifstream& inf, int Batch_Size,float *&wei,float *&tmp)
	{
		int INS,OUS;
		readf(inf,INS),readf(inf,OUS);
		int nou=0;
		init(nou,Batch_Size,INS,OUS);
		initmem(wei,tmp);
	}
	inline void forward(int Batch_Size,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out,
						bool test)
	{
		assert(Batch_Size==bs&&ins==id*ih*iw&&ous==od*oh*ow);
		if(test) memset(out,0,sizeof(float) * ous);
		else memset(out, 0, sizeof(float) * bs * ous);
		for(int tb=0;tb<bs;tb++)
		{
			int adi=tb*ins,ado=tb*ous;
			for (int i = 0; i < ins; i++) for (int j = 0; j < ous; j++) out[ado+j] += in[adi+i] * w[i * ous + j];
			if(test) break;
		}
	}
	inline void backward(int Batch_Size,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		assert(Batch_Size==bs&&ins==id*ih*iw&&ous==od*oh*ow);
		memset(din, 0, sizeof(float) * bs * ins);
		for(int tb=0;tb<bs;tb++)
		{
			int adi=tb*ins,ado=tb*ous;
			for (int i = 0; i < ous; i++)
			{
				float pre = dout[ado+i];
				for (int j = 0; j < ins; j++)
				{
					din[adi+j] += pre * w[j * ous + i];
					tmpw[j * ous + i] += in[adi+j] * pre;
				}
			}
		}
	}
	inline val3d operator()(val3d x)
	{
		val3d res(ous,1,1);
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
		AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(FC)
	#endif
};
