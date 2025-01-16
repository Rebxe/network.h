#pragma once

#include "../defines.h"

// Convolution layer

class CONV
{
public:
	int bs, ind, inh, inw;
	int cnt, ch, cw;
	int stx, sty;
	int pdx, pdy;
	float pdval;
	int ouh, ouw;
	float * w;

private:
	float * tmpw;
	inline void initmem(float *&wei,float *&tmp)
	{
		w=wei,tmpw=tmp;
		wei += ind * ch * cw  * cnt, tmp += ind * ch * cw * cnt;
	}
	#ifndef ENABLE_OPENMP
		float * inmrt, * dinmrt;
	#else
		float * inmrt[THREAD_NUM], * dinmrt[THREAD_NUM], * ttmpw[THREAD_NUM];
	#endif
	inline void getmem()
	{
		#ifndef ENABLE_OPENMP
			inmrt = new float[ouh * ouw * ind * ch * cw];
			dinmrt = new float[ouh * ouw * ind * ch * cw];
		#else
			for(int i=0;i<THREAD_NUM;i++)
			{
				inmrt[i] = new float[ouh * ouw * ind * ch * cw];
				dinmrt[i] = new float[ouh * ouw * ind * ch * cw];
				ttmpw[i] = new float[ind * ch * cw * cnt];
			}
		#endif
	}
	inline void freemem()
	{
		#ifndef ENABLE_OPENMP
			delete[] inmrt;
			delete[] dinmrt;
		#else
			for(int i=0;i<THREAD_NUM;i++)
			{
				delete[] inmrt[i];
				delete[] dinmrt[i];
				delete[] ttmpw[i];
			}
		#endif
	}

public:
	inline void init(int &m, int Batch_Size, 
		SHAPE3D Input,
		int CoreCnt, std::pair<int,int> Core,
		std::pair<int,int> Stride = {1,1},
		std::pair<int,int> Padding = {0,0}, float PaddingVal = 0)
	{
		bs = Batch_Size;
		ind=std::get<0>(Input),inh=std::get<1>(Input),inw=std::get<2>(Input),
		cnt = CoreCnt, ch = Core.first, cw = Core.second;
		stx = Stride.first, sty = Stride.second;
		pdx = Padding.first, pdy = Padding.second, pdval = PaddingVal;
		ouh = (inh + pdx * 2 - ch) / stx + 1, ouw = (inw + pdy * 2 - cw) / sty + 1;
		m += ind * ch * cw * cnt;
	}
	inline void build(float *&wei,float *&tmp,int InitType = INIT_HE)
	{
		initmem(wei,tmp);
		// init w
		std::default_random_engine gen;
		std::normal_distribution<float> wer;
		if (InitType == INIT_HE) wer = std::normal_distribution<float>(0, sqrt(2 / (float)ind / ch / cw));
		else                     wer = std::normal_distribution<float>(0, sqrt(1 / (float)ind / ch / cw));
		gen.seed(time(NULL));
		for (int i = 0; i < ind * ch * cw * cnt; i++) w[i] = wer(gen);
	}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,(SHAPE3D){ind,inh,inw});
		writf(ouf,cnt),writf(ouf,std::make_pair(ch,cw));
		writf(ouf,std::make_pair(stx,sty));
		writf(ouf,std::make_pair(pdx,pdy)),writf(ouf,pdval);
	}
	inline void load(std::ifstream& inf, int Batch_Size,float *&wei, float *&tmp)
	{
		SHAPE3D Input;
		int CoreCnt;
		std::pair<int,int> Core,Stride,Padding;
		float PaddingVal;
		readf(inf,Input);
		readf(inf,CoreCnt),readf(inf,Core);
		readf(inf,Stride);
		readf(inf,Padding),readf(inf,PaddingVal);
		int nou=0;
		init(nou,Batch_Size,
			Input,
			CoreCnt,Core,
			Stride,
			Padding,PaddingVal);
		initmem(wei,tmp);
	}

private:
	inline void forward(int Batch_Size,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		if(Batch_Size!=0) assert(Batch_Size==bs);
		assert(ind==id&&inh==ih&&inw==iw&&cnt==od&&ouh==oh&&ouw==ow);
		getmem();
		if(Batch_Size!=0)
		{
			#ifdef ENABLE_OPENMP
				#pragma omp parallel for schedule(static) num_threads(THREAD_NUM)
			#endif
			for(int tb=0;tb<bs;tb++)
			{
				float *_inmrt;
				#ifndef ENABLE_OPENMP
					_inmrt=inmrt;
				#else
					_inmrt=inmrt[omp_get_thread_num()];
				#endif
				int adin = tb * ind * inh * inw;
				int adout = tb * cnt * ouh * ouw;
				img2col(ind,inh,inw,ch,cw,stx,sty,pdx,pdy,pdval,in + adin,_inmrt);
				Matrix_Mul(ouh * ouw, ind * ch * cw, cnt, _inmrt, w, out + adout);
			}
		}
		else
		{
			float *_inmrt;
			#ifndef ENABLE_OPENMP
				_inmrt=inmrt;
			#else
				_inmrt=inmrt[0];
			#endif
			img2col(ind,inh,inw,ch,cw,stx,sty,pdx,pdy,pdval,in,_inmrt);
			Matrix_Mul(ouh * ouw, ind * ch * cw, cnt, _inmrt, w, out);
		}
		freemem();
	}
	inline void backward(int Batch_Size,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		assert(Batch_Size==bs&&ind==id&&inh==ih&&inw==iw&&cnt==od&&ouh==oh&&ouw==ow);
		getmem();
		memset(din,0,sizeof(float)*bs*ind*inh*inw);
		#ifdef ENABLE_OPENMP
			for(int i=0;i<THREAD_NUM;i++) memset(ttmpw[i],0,sizeof(float)*ind * ch * cw * cnt);
			#pragma omp parallel for schedule(static) num_threads(THREAD_NUM)
		#endif
		for(int tb=0;tb<bs;tb++)
		{
			float *_dinmrt,*_inmrt,*_tmpw;
			#ifndef ENABLE_OPENMP
				_dinmrt=dinmrt;
				_inmrt=inmrt;
				_tmpw=tmpw;
			#else
				_dinmrt=dinmrt[omp_get_thread_num()]; 
				_inmrt=inmrt[omp_get_thread_num()];
				_tmpw=ttmpw[omp_get_thread_num()];
			#endif
			int adin = tb * ind * inh * inw;
			int adout = tb * cnt * ouh * ouw;
			// upd d_in
			Matrix_Mul_Back_A(ouh * ouw, ind * ch * cw, cnt, _dinmrt, w, dout + adout);
			col2img(ind,inh,inw,ch,cw,stx,sty,pdx,pdy,_dinmrt,din + adin);
			// upd d_w
			img2col(ind,inh,inw,ch,cw,stx,sty,pdx,pdy,pdval,in + adin,_inmrt);
			Matrix_Mul_Back_B(ouh * ouw, ind * ch * cw, cnt, _inmrt, _tmpw, dout + adout);
		}
		#ifdef ENABLE_OPENMP
			for(int i=0;i<THREAD_NUM;i++)
				for(int j=0;j<ind * ch * cw * cnt;j++) tmpw[j]+=ttmpw[i][j];
		#endif
		freemem();
	}

public:
	inline val3d operator()(val3d x)
	{
		val3d res(cnt,ouh,ouw);
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
		AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(CONV)
	#endif
};
