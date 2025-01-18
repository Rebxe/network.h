#pragma once

#include "../defines.h"

// Deconvolution layer

class DECONV
{
public:
	int ind, inh, inw;
	int cnt, ch, cw;
	int stx, sty;
	int pdx, pdy;
	int ouh, ouw;
	float * w;

private:
	float * tmpw;
	inline void initmem(float *wei, float *tmp)
	{
		w = wei, tmpw = tmp;
		wei += cnt * ch * cw * ind, tmp += cnt * ch * cw * ind;
	}
	#ifndef ENABLE_OPENMP
		float * outmrt;
	#else
		float * outmrt[THREAD_NUM], * ttmpw[THREAD_NUM];
	#endif
	inline void getmem()
	{
		#ifndef ENABLE_OPENMP 
			outmrt = new float[inh * inw * cnt * ch * cw];
		#else
			for(int i=0;i<THREAD_NUM;i++)
			{
				outmrt[i] = new float[inh * inw * cnt * ch * cw];
				ttmpw[i] = new float[cnt * ch * cw * ind];
			}
		#endif
	}
	inline void freemem()
	{
		#ifndef ENABLE_OPENMP
			delete[] outmrt;
		#else
			for(int i=0;i<THREAD_NUM;i++)
			{
				delete[] outmrt[i];
				delete[] ttmpw[i];
			}
		#endif
	}

public:
	inline void init(int &m, 
		SHAPE3D Input,
		int CoreCnt, std::pair<int,int> Core,
		std::pair<int,int> Stride = {1,1},
		std::pair<int,int> Padding = {0,0})
	{
		ind = std::get<0>(Input), inh = std::get<1>(Input), inw = std::get<2>(Input);
		cnt = CoreCnt, ch = Core.first, cw = Core.second;
		stx = Stride.first, sty = Stride.second;
		pdx = Padding.first, pdy = Padding.second;
		ouh = ch + (inh - 1) * stx - pdx * 2, ouw = cw + (inw - 1) * sty - pdy * 2;
		m += cnt * ch * cw * ind;
	}
	inline void build(float *&wei,float *&tmp,int InitType = INIT_HE)
	{
		initmem(wei,tmp);
		// init w
		std::default_random_engine gen;
		std::normal_distribution<float> wer;
		int ins = ind * ((ch + stx - 1) / stx + (cw + sty - 1) / sty);
		if (InitType == INIT_HE) wer = std::normal_distribution<float>(0, sqrt(2 / (float)ins));
		else                     wer = std::normal_distribution<float>(0, sqrt(1 / (float)ins));
		gen.seed(time(NULL));
		for (int i = 0; i < cnt * ch * cw * ind; i++) w[i] = wer(gen);
	}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,(SHAPE3D){ind,inh,inw});
		writf(ouf,cnt),writf(ouf,std::make_pair(ch,cw));
		writf(ouf,std::make_pair(stx,sty));
		writf(ouf,std::make_pair(pdx,pdy));
	}
	inline void load(std::ifstream& inf,float *&wei, float *&tmp)
	{
		SHAPE3D Input;
		int CoreCnt;
		std::pair<int,int> Core,Stride,Padding;
		readf(inf,Input);
		readf(inf,CoreCnt),readf(inf,Core);
		readf(inf,Stride),readf(inf,Padding);
		int nou=0;
		init(nou,
			Input,
			CoreCnt,Core,
			Stride,
			Padding);
		initmem(wei,tmp);
	}

private: 
	inline void forward(int bs,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		ext_assert(ind==id&&inh==ih&&inw==iw&&cnt==od&&ouh==oh&&ouw==ow,
			fprintf(stderr,"\
In DECONV::forward(...)\n\
  in  = [%d * %d * %d]\n\
  out = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",ind,inh,inw,cnt,ouh,ouw,id,ih,iw,od,oh,ow));
		getmem();
		if(bs!=0)
		{
			memset(out,0,sizeof(float)*bs*cnt*ouh*ouw);
			#ifdef ENABLE_OPENMP
				#pragma omp parallel for schedule(static) num_threads(THREAD_NUM)
			#endif
			for(int tb=0;tb<bs;tb++)
			{
				float *_outmrt;
				#ifndef ENABLE_OPENMP
					_outmrt=outmrt;
				#else
					_outmrt=outmrt[omp_get_thread_num()];
				#endif
				int adin = tb * ind * inh * inw;
				int adout = tb * cnt * ouh * ouw;
				Matrix_Mul_Back_A(inh * inw, cnt * ch * cw, ind, _outmrt, w, in + adin);
				col2img(cnt,ouh,ouw,ch,cw,stx,sty,pdx,pdy,_outmrt,out+adout);
			}
		}
		else
		{
			memset(out,0,sizeof(float)*cnt*ouh*ouw);
			float *_outmrt;
			#ifndef ENABLE_OPENMP
				_outmrt=outmrt;
			#else
				_outmrt=outmrt[0];
			#endif
			Matrix_Mul_Back_A(inh * inw, cnt * ch * cw, ind, _outmrt, w, in);
			col2img(cnt,ouh,ouw,ch,cw,stx,sty,pdx,pdy,_outmrt,out);
		}
		freemem();
	}
	inline void backward(int bs,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		ext_assert(ind==id&&inh==ih&&inw==iw&&cnt==od&&ouh==oh&&ouw==ow,
			fprintf(stderr,"\
In DECONV::backward(...)\n\
  in  = [%d * %d * %d]\n\
  out = [%d * %d * %d]\n\
but\n\
  Real Input  = [%d * %d * %d]\n\
  Real Output = [%d * %d * %d]\n\n",ind,inh,inw,cnt,ouh,ouw,id,ih,iw,od,oh,ow));
		getmem();
		memset(din,0,sizeof(float)*bs*ind*inh*inw);
		#ifdef ENABLE_OPENMP
			for(int i=0;i<THREAD_NUM;i++) memset(ttmpw[i],0,sizeof(float)*cnt * ch * cw * ind);
			#pragma omp parallel for schedule(static) num_threads(THREAD_NUM)
		#endif
		for(int tb=0;tb<bs;tb++)
		{
			float *_doutmrt,*_tmpw;
			#ifndef ENABLE_OPENMP
				_doutmrt=outmrt;
				_tmpw=tmpw;
			#else
				_doutmrt=outmrt[omp_get_thread_num()];
				_tmpw=ttmpw[omp_get_thread_num()];
			#endif
			int adin = tb * ind * inh * inw;
			int adout = tb * cnt * ouh * ouw;
			img2col(cnt,ouh,ouw,ch,cw,stx,sty,pdx,pdy,(float)0,dout+adout,_doutmrt);
			// upd d_in
			Matrix_Mul(inh * inw, cnt * ch * cw, ind, _doutmrt, w, din + adin);
			// upd d_w
			Matrix_Mul_Back_B(inh * inw, cnt * ch * cw, ind, _doutmrt, _tmpw, in + adin);
		}
		#ifdef ENABLE_OPENMP
			for(int i=0;i<THREAD_NUM;i++)
				for(int j=0;j<cnt * ch * cw * ind;j++) tmpw[j]+=ttmpw[i][j];
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
		AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(DECONV)
	#endif
};
