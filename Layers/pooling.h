#pragma once

#include "../defines.h"

// Pooling Layer

#define MAX_POOLING    1
#define MEAN_POOLING   2

class POOLING
{
public:
	int ind, inh, inw;
	int ch, cw;
	int tpe;
	int stx, sty;
	int ouh, ouw;

public:
	inline void init(SHAPE3D Input, 
		std::pair<int,int> Core, 
		int Type = MAX_POOLING, 
		std::pair<int,int> Stride = {-1,-1}) // -1 : equal Core
	{
		ind = std::get<0>(Input), inh = std::get<1>(Input), inw = std::get<2>(Input);
		ch = Core.first, cw = Core.second;
		tpe = Type;
		stx = Stride.first == -1 ? ch : Stride.first, sty = Stride.second == -1 ? cw : Stride.second;
		ouh = (inh + stx - 1) / stx, ouw = (inw + sty - 1) / sty;
	}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,(SHAPE3D){ind,inh,inw});
		writf(ouf,std::make_pair(ch,cw));
		writf(ouf,tpe);
		writf(ouf,std::make_pair(stx,sty));
	}
	inline void load(std::ifstream& inf)
	{
		SHAPE3D Input;
		std::pair<int,int> Core;
		int Type;
		std::pair<int,int> Stride;
		readf(inf,Input),readf(inf,Core),readf(inf,Type),readf(inf,Stride);
		init(Input,Core,Type,Stride);
	}

private:
	inline void forward(int bs,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		ext_assert(ind==id&&inh==ih&&inw==iw&&ind==od&&ouh==oh&&ouw==ow,
			fprintf(stderr,"\
In POOLING::forward(...)\n\
  in  = [%d * %d * %d]\n\
  out = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",ind,inh,inw,ind,ouh,ouw,id,ih,iw,od,oh,ow));
		bs=std::max(bs,1);
		for(int tb=0;tb<bs;tb++)
		{
			int adi=tb*ind*inh*inw,ado=tb*ind*ouh*ouw;
			for (int d = 0; d < ind; d++)
			{
				int ad = d * inh * inw;
				for (int i = 0, id = 0; i < ouh * stx; i += stx, id++)
					for (int j = 0, jd = 0; j < ouw * sty; j += sty, jd++)
					{
						float& pre = out[ado + d * ouh * ouw + id * ouw + jd];
						if(tpe==MAX_POOLING) pre = -1e8;
						if(tpe==MEAN_POOLING) pre = 0;
						int tot = 0;
						for (int x = i; x < i + ch && x < inh; x++)
							for (int y = j; y < j + cw && y < inw; y++)
							{
								if (tpe == MAX_POOLING) pre = (std::max)(pre, in[adi + ad + x * inw + y]);
								if (tpe == MEAN_POOLING) pre += in[adi + ad + x * inw + y];
								tot++;
							}
						if (tpe == MEAN_POOLING) pre /= tot;
					}
			}
		}
	}
	inline void backward(int bs,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		ext_assert(ind==id&&inh==ih&&inw==iw&&ind==od&&ouh==oh&&ouw==ow,
			fprintf(stderr,"\
In POOLING::backward(...)\n\
  in  = [%d * %d * %d]\n\
  out = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",ind,inh,inw,ind,ouh,ouw,id,ih,iw,od,oh,ow));
		memset(din, 0, sizeof(float) * bs * ind * inh * inw);
		for(int tb=0;tb<bs;tb++)
		{
			int adi=tb*ind*inh*inw,ado=tb*ind*ouh*ouw;
			for (int d = 0; d < ind; d++)
			{
				int ad = d * inh * inw;
				for (int i = 0, id = 0; i < ouh * stx; i += stx, id++)
					for (int j = 0, jd = 0; j < ouw * sty; j += sty, jd++)
					{
						float& pre = dout[ado + d * ouh * ouw + id * ouw + jd], pout;
						int tot = ((std::min)(i + ch, inh) - i) * ((std::min)(j + cw, inw) - j);
						if(tpe == MAX_POOLING)
						{
							pout = -1e8;
							for (int x = i; x < i + ch && x < inh; x++)
								for (int y = j; y < j + cw && y < inw; y++)
									pout = (std::max)(pout, in[adi + ad + x * inw + y]);
						}
						for (int x = i; x < i + ch && x < inh; x++)
							for (int y = j; y < j + cw && y < inw; y++)
							{
								if (tpe == MAX_POOLING && fabs(pout - in[adi + ad + x * inw + y]) < DBL_EPSILON) din[adi + ad + x * inw + y] += pre;
								if (tpe == MEAN_POOLING) din[adi + ad + x * inw + y] += pre / tot;
							}
					}
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
		AUTO_SL_LAYER_CONSTRUCTER(POOLING)
	#endif
};
