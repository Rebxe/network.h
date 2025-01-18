#pragma once

#include "../defines.h"


class SGD
{
public:
	int m; // Count of weights
	float lrt;

private:
	float *wei,*tmp;
	inline void initmem(){wei=new float[m],tmp=new float[m];}

public:
	inline void init(float Learn_Rate)
	{
		lrt=Learn_Rate;
		m=0;
	}
	inline void build(){initmem();}
	inline void save(std::ofstream& ouf)
	{
		writf(ouf,m),writf(ouf,lrt);
		writf(ouf,wei,m);
	}
	inline void load(std::ifstream& inf)
	{
		readf(inf,m),readf(inf,lrt);
		initmem();
		readf(inf,wei,m);
	}
	inline void delthis(){delete[] wei,delete[] tmp;}
	inline float* _wei(){return wei;}
	inline float* _tmp(){return tmp;}
	inline void init_backward(){memset(tmp,0,sizeof(float)*m);}
	inline void flush(){for(int i=0;i<m;i++) wei[i]-=lrt*tmp[i];}
	#ifdef ENABLE_AUTO_SL
		AUTO_SL_OPTIMIZER_CONSTRUCTER(SGD)
	#endif
};
