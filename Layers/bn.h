#pragma once

#include "../defines.h"

// Batch Normalization Layer

class BN
{
public:
	int bs,d,h,w;
	float delta, eps; // avg_x=avg_x*delta+pre_x*(1-delta)
	float *k,*b;
	float *e_avg,*e_var; // Expect_avg, Expect_variance
	
private:
	float *t_avg,*t_var;
	float *tmpk,*tmpb;
	inline void initmem(float *&wei, float *&tmp)
	{
		t_avg=new float[d], t_var=new float[d];
		e_avg=new float[d], e_var=new float[d];
		k=wei,tmpk=tmp;
		wei+=d,tmp+=d;
		b=wei,tmpb=tmp;
		wei+=d,tmp+=d;
	}

public:
	inline void init(int &m, int Batch_Size,SHAPE3D Input,float Delta = 0.9, float EPS = 1e-4)
	{
		bs=Batch_Size;
		d=std::get<0>(Input),h=std::get<1>(Input),w=std::get<2>(Input);
		delta=Delta,eps=EPS;
		m+=d,m+=d;
	}
	inline void build(float *&wei, float *&tmp)
	{
		initmem(wei,tmp);
		for(int i=0;i<d;i++) k[i]=1,b[i]=0,e_avg[i]=0,e_var[i]=1;
	}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,(SHAPE3D){d,h,w});
		writf(ouf,delta),writf(ouf,eps);
		writf(ouf,e_avg,d),writf(ouf,e_var,d);
	}
	inline void load(std::ifstream& inf, int Batch_Size, float *&wei, float *&tmp)
	{
		SHAPE3D Input;
		float Delta,EPS;
		readf(inf,Input);
		readf(inf,Delta),readf(inf,EPS);
		int nou=0;
		init(nou,Batch_Size,Input,Delta,EPS);
		initmem(wei,tmp);
		readf(inf,e_avg,d),readf(inf,e_var,d);
	}
	inline void delthis() { delete[] t_avg,delete[] t_var,delete[] e_avg,delete[] e_var; }
	inline void forward(int Batch_Size,
						int id,int ih,int iw,float *in,
						int od,int oh,int ow,float *out,
						bool test)
	{
		assert(Batch_Size==bs&&d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow);
		for(int i=0;i<d;i++)
		{
			int siz=d*h*w,ad=i*h*w;
			if(!test)
			{
				float &avg=t_avg[i],&var=t_var[i];
				avg=var=0;
				for(int j=0;j<bs;j++) for(int t=0;t<h*w;t++) avg+=in[j*siz+ad+t];
				avg/=bs*h*w;
				for(int j=0;j<bs;j++) for(int t=0;t<h*w;t++) var+=pow(in[j*siz+ad+t]-avg,2);
				var/=bs*h*w;
				float s=sqrt(var+eps);
				for(int j=0;j<bs;j++)
					for(int t=0;t<h*w;t++)
						out[j*siz+ad+t]=k[i]*(in[j*siz+ad+t]-avg)/s+b[i];
				e_avg[i]=e_avg[i]*delta+avg*(1-delta);
				e_var[i]=e_var[i]*delta+var*(1-delta);
			}
			else
			{
				float s=sqrt(e_var[i]+eps);
				for(int t=0;t<h*w;t++) out[ad+t]=k[i]*(in[ad+t]-e_avg[i])/s+b[i];
			}
		}
	}
	inline void backward(int Batch_Size,
						 int id,int ih,int iw,float *in, float* din,
						 int od,int oh,int ow,float *dout)
	{
		assert(Batch_Size==bs&&d==id&&h==ih&&w==iw&&d==od&&h==oh&&w==ow);
		for(int i=0;i<d;i++)
		{
			int siz=d*h*w,ad=i*h*w;
			float avg=t_avg[i],var=t_var[i];
			float s=sqrt(var+eps),s2=pow(var+eps,1.5)*2;
			for(int j=0;j<bs;j++)
				for(int t=0;t<h*w;t++) tmpk[i]+=dout[j*siz+ad+t]*(in[j*siz+ad+t]-avg)/s;
			for(int j=0;j<bs;j++)
				for(int t=0;t<h*w;t++) tmpb[i]+=dout[j*siz+ad+t];
			float dL_davg=0,dL_dvar=0,dvar_davg=0;
			for(int j=0;j<bs;j++)
				for(int t=0;t<h*w;t++)
				{
					int id=j*siz+ad+t;
					float dL_dhx=dout[id]*k[i];
					dL_davg-=(din[id]=dL_dhx/s);
					dL_dvar-=dL_dhx*(in[id]-avg)/s2;
					dvar_davg+=2/(float)(bs*h*w)*(avg-in[id]);
				}
			dL_davg+=dL_dvar*dvar_davg;
			for(int j=0;j<bs;j++)
				for(int t=0;t<h*w;t++)
				{
					int id=j*siz+ad+t;
					din[id]+=(dL_davg+dL_dvar*2*(in[id]-avg))/(float)(bs*h*w);
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
		AUTO_SL_LAYER_CONSTRUCTER_WEIGHT_DELTHISFUNC(BN)
	#endif
};
