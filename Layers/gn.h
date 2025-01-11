#pragma once

#include "../defines.h"

// Group Normalization Layer

class GN
{
public:
	int bs,d,h,w;
	int g;
	float eps;
	int cnt; // number of groups
	float *k,*b;
	
private:
	float *tmpk,*tmpb;
	inline void initmem(float *&wei, float *&tmp)
	{
		k=wei,tmpk=tmp;
		wei+=cnt,tmp+=cnt;
		b=wei,tmpb=tmp;
		wei+=cnt,tmp+=cnt;
	}

public:
	inline void init(int &m, int Batch_Size,SHAPE3D Input,int G,float Eps=1e-4)
	{
		bs=Batch_Size;
		d=std::get<0>(Input),h=std::get<1>(Input),w=std::get<2>(Input);
		g=G,eps=Eps;
		cnt=d/g+(d%g!=0);
		m+=cnt,m+=cnt;
	}
	inline void build(float *&wei, float *&tmp)
	{
		initmem(wei,tmp);
		for(int i=0;i<cnt;i++) k[i]=1,b[i]=0;
	}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,(SHAPE3D){d,h,w});
		writf(ouf,g),writf(ouf,eps);
	}
	inline void load(std::ifstream& inf, int Batch_Size, float *&wei, float *&tmp)
	{
		SHAPE3D Input;
		int G;
		float Eps;
		readf(inf,Input);
		readf(inf,G),readf(inf,Eps);
		int nou=0;
		init(nou,Batch_Size,Input,G,Eps);
		initmem(wei,tmp);
	}
	inline void forward(int Batch_Size,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out)
	{
		if(Batch_Size!=0) assert(Batch_Size==bs);
		assert(d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow);
		for(int tb=0;tb<std::max(Batch_Size,1);tb++)
		{
			int ad=tb*d*h*w; 
			for(int l=0,id=0;l<d;l+=g,id++)
			{
				int r=std::min(d-1,l+g-1);
				float avg=0,var=0;
				for(int j=l;j<=r;j++) for(int t=0;t<h*w;t++) avg+=in[ad+j*h*w+t];
				avg/=(r-l+1)*h*w;
				for(int j=l;j<=r;j++) for(int t=0;t<h*w;t++) var+=pow(in[ad+j*h*w+t]-avg,2);
				var/=(r-l+1)*h*w;
				float s=sqrt(var+eps);
				for(int j=l;j<=r;j++)
					for(int t=0;t<h*w;t++) out[ad+j*h*w+t]=k[id]*(in[ad+j*h*w+t]-avg)/s+b[id];
			}
		}
	}
	inline void backward(int Batch_Size,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		assert(Batch_Size==bs&&d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow);
		for(int tb=0;tb<bs;tb++)
		{
			int ad=tb*d*h*w; 
			for(int l=0,id=0;l<d;l+=g,id++)
			{
				int r=std::min(d-1,l+g-1);
				float avg=0,var=0;
				for(int j=l;j<=r;j++) for(int t=0;t<h*w;t++) avg+=in[ad+j*h*w+t];
				avg/=(r-l+1)*h*w;
				for(int j=l;j<=r;j++) for(int t=0;t<h*w;t++) var+=pow(in[ad+j*h*w+t]-avg,2);
				var/=(r-l+1)*h*w;
				float s=sqrt(var+eps),s2=pow(var+eps,1.5)*2;
				for(int j=l;j<=r;j++)
					for(int t=0;t<h*w;t++) tmpk[id]+=dout[ad+j*h*w+t]*(in[ad+j*h*w+t]-avg)/s;
				for(int j=l;j<=r;j++) for(int t=0;t<h*w;t++) tmpb[id]+=dout[ad+j*h*w+t];
				float dL_davg=0,dL_dvar=0,dvar_davg=0;
				for(int j=l;j<=r;j++)
					for(int t=0;t<h*w;t++)
					{
						int idx=ad+j*h*w+t;
						float dL_dhx=dout[idx]*k[id];
						dL_davg-=(din[idx]=dL_dhx/s);
						dL_dvar-=dL_dhx*(in[idx]-avg)/s2;
						dvar_davg+=2/(float)((r-l+1)*h*w)*(avg-in[idx]);
					}
				dL_davg+=dL_dvar*dvar_davg;
				for(int j=l;j<=r;j++)
					for(int t=0;t<h*w;t++)
					{
						int idx=ad+j*h*w+t;
						din[idx]+=(dL_davg+dL_dvar*2*(in[idx]-avg))/(float)((r-l+1)*h*w);
					}
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
		AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(GN)
	#endif
};
