#pragma once

#include "../defines.h"


class ADAM
{
public:
	int m; // Batch_Size, Count of weights
	float lrt,b1,b2,eps;
	
private:
	float *wei,*tmp;
	float mb1,mb2;
	float *v,*s;
	inline void initmem()
	{
		wei=new float[m],tmp=new float[m];
		v=new float[m],s=new float[m];
	}

public:
	inline void init(float Learn_Rate,float beta1=0.9,float beta2=0.999,float Eps=1e-8)
	{
		lrt=Learn_Rate;
		b1=beta1,b2=beta2,eps=Eps;
		m=0;
	}
	inline void build()
	{
		initmem(); 
		mb1=1,mb2=1;
		memset(v,0,sizeof(float)*m),memset(s,0,sizeof(float)*m);
	}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,m),writf(ouf,lrt);
		writf(ouf,b1),writf(ouf,b2),writf(ouf,eps);
		writf(ouf,mb1),writf(ouf,mb2);
		writf(ouf,wei,m);
		writf(ouf,v,m),writf(ouf,s,m);
	}
	inline void load(std::ifstream& inf)
	{
		readf(inf,m),readf(inf,lrt);
		initmem();
		readf(inf,b1),readf(inf,b2),readf(inf,eps);
		readf(inf,mb1),readf(inf,mb2);
		readf(inf,wei,m);
		readf(inf,v,m),readf(inf,s,m);
	}
	inline void delthis(){delete[] wei,delete[] tmp,delete[] v,delete[] s;}
	inline float* _wei(){return wei;}
	inline float* _tmp(){return tmp;}
	inline void init_backward(){memset(tmp,0,sizeof(float)*m);}
	inline void flush()
	{
		mb1*=b1,mb2*=b2;
		for(int i=0;i<m;i++)
		{
			float g=tmp[i];
			v[i]=b1*v[i]+(1-b1)*g;
			s[i]=b2*s[i]+(1-b2)*g*g;
			float vh=v[i]/(1-mb1),sh=s[i]/(1-mb2); 
			wei[i]-=lrt*vh/(sqrt(sh)+eps);
		}
	}
	#ifdef ENABLE_AUTO_SL
		AUTO_SL_OPTIMIZER_CONSTRUCTER(ADAM)
	#endif
};
