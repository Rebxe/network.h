#pragma once

#include <assert.h>
#include <functional>
#include <vector>
#include <cstring>
#include <cmath>

namespace auto_dao
{
	int Batch_Size=0;
	bool test=false;
	
	struct node
	{
		int d,h,w;
		float *a,*da;
		node *in1,*in2;
		int oud,cnt;
		std::function<void(int,
						   int,int,int,float*,
						   int,int,int,float*,
						   bool)>                forward_f; // in out test
		std::function<void(int,
						   int,int,int,float*,float*,
						   int,int,int,float*)> backward_f; // in din dout
		std::function<void(int,
						   int,int,int,float*,
						   int,int,int,float*,
						   int,int,int,float*,
						   bool)>               forward_f2; // in1 in2 out test
		std::function<void(int,
						   int,int,int,float*,float*,
						   int,int,int,float*,float*,
						   int,int,int,float*)> backward_f2;// in1 din1 in2 din2 dout
		inline node(int td,int th,int tw);
		inline ~node();
		inline void forward();
		inline void backward();
	};
	
	std::vector<node*> tmp;
	
	inline void init(int BatchSize,bool flg_test)
	{
		Batch_Size=BatchSize;
		test=flg_test;
		for(node *x:tmp) delete x;
		tmp.clear();
	}
	inline void init_backward(){for(node *x:tmp) x->cnt=0;}
}

class val3d
{
	public:
		int d,h,w;
		float *a,*da;
		inline val3d(){}
		inline val3d(int td,int th,int tw,float val=0);
		inline val3d(int td,int th,int tw,float *dat);
		inline void backward();

	public:
		auto_dao::node *dat;
};
		
inline val3d reshape(val3d x,int d,int h,int w);
inline val3d toshape(val3d x,int d,int h,int w);
inline val3d operator+(val3d x,val3d y);
inline val3d operator-(val3d x,val3d y);
inline val3d operator*(val3d x,val3d y);
inline val3d operator/(val3d x,val3d y);
inline val3d dcat(val3d x,val3d y);

inline float MSEloss(val3d x,float *realout);
inline float BCEloss(val3d x,float *realout); 

/***************************** End of definitions *****************************/

inline auto_dao::node::node(int td,int th,int tw)
{
	assert(Batch_Size!=0);
	d=td,h=th,w=tw;
	a=new float[Batch_Size*d*h*w],da=new float[Batch_Size*d*h*w];
	in1=in2=0;
	oud=cnt=0;
	tmp.push_back(this);
}

inline auto_dao::node::~node(){delete[] a,delete[] da;}

inline void auto_dao::node::forward()
{
	if(in1==0&&in2==0) return;
	if(in2==0) forward_f(Batch_Size,
						 in1->d,in1->h,in1->w,in1->a,
						 d,h,w,a,
						 test);
	else
	{
		forward_f2(Batch_Size,
				   in1->d,in1->h,in1->w,in1->a,
				   in2->d,in2->h,in2->w,in2->a,
				   d,h,w,a,
				   test);
	}
}

inline void auto_dao::node::backward()
{
	if(in2==0&&in1==0) return;
	if(in2==0)
	{
		if(in1->cnt==0) backward_f(Batch_Size,
						 		   in1->d,in1->h,in1->w,in1->a,in1->da,
						 		   d,h,w,da);
		else
		{
			float *tmpda=new float[Batch_Size*(in1->d)*(in1->h)*(in1->w)];
			backward_f(Batch_Size,
		 		   	   in1->d,in1->h,in1->w,in1->a,tmpda,
		 		       d,h,w,da);
			for(int i=0;i<Batch_Size*(in1->d)*(in1->h)*(in1->w);i++)
				in1->da[i]+=tmpda[i];
			delete[] tmpda;
		}
		if(++in1->cnt==in1->oud) in1->backward();
	}
	else
	{
		bool f1,f2;
		float *d1,*d2;
		if(in1->cnt==0) f1=true,d1=in1->da;
		else f1=false,d1=new float[Batch_Size*(in1->d)*(in1->h)*(in1->w)];
		if(in2->cnt==0) f2=true,d2=in2->da;
		else f2=false,d2=new float[Batch_Size*(in2->d)*(in2->h)*(in2->w)];
		backward_f2(Batch_Size,
				    in1->d,in1->h,in1->w,in1->a,d1,
				    in2->d,in2->h,in2->w,in2->a,d2,
				    d,h,w,da);
		if(!f1)
		{
			for(int i=0;i<Batch_Size*(in1->d)*(in1->h)*(in1->w);i++)
				in1->da[i]+=d1[i];
			delete[] d1;
		}
		if(!f2)
		{
			for(int i=0;i<Batch_Size*(in2->d)*(in2->h)*(in2->w);i++)
				in2->da[i]+=d2[i];
			delete[] d2;
		}
		if(++in1->cnt==in1->oud) in1->backward();
		if(++in2->cnt==in2->oud) in2->backward();
	}
}

inline val3d::val3d(int td,int th,int tw,float val)
{
	d=td,h=th,w=tw;
	dat=new auto_dao::node(td,th,tw);
	a=dat->a;
	da=dat->da;
	int bs=auto_dao::test?1:auto_dao::Batch_Size;
	for(int i=0;i<bs*d*h*w;i++) a[i]=val;
}

inline val3d::val3d(int td,int th,int tw,float *val)
{
	d=td,h=th,w=tw;
	dat=new auto_dao::node(td,th,tw);
	a=dat->a;
	da=dat->da;
	int bs=auto_dao::test?1:auto_dao::Batch_Size;
	for(int i=0;i<bs*d*h*w;i++) a[i]=val[i];
}

inline void val3d::backward(){dat->backward();}

inline val3d reshape(val3d x,int d,int h,int w)
{
	assert(x.d*x.h*x.w==d*h*w);
	val3d res(d,h,w);
	res.dat->in1=x.dat;
	x.dat->oud++;
	res.dat->forward_f=[&](int bs,
	   int d1,int h1,int w1,float* a,
	   int d2,int h2,int w2,float* res,
	   bool test)
    {
        assert(bs==auto_dao::Batch_Size&&d1*h1*w1==d2*h2*w2);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) res[ad+j]=a[ad+j];
			if(test) break;
		}
    };
	res.dat->backward_f=[&](int bs,
	   int d1,int h1,int w1,float* a,float *da,
	   int d2,int h2,int w2,float* dres)
    {
        assert(bs==auto_dao::Batch_Size&&d1*h1*w1==d2*h2*w2);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) da[ad+j]=dres[ad+j];
		}
    };
    res.dat->forward();
    return res;
}

inline val3d toshape(val3d x,int d,int h,int w)
{
	val3d res(d,h,w);
	res.dat->in1=x.dat;
	x.dat->oud++;
	res.dat->forward_f=[&](int bs,
	   int d1,int h1,int w1,float* a,
	   int d2,int h2,int w2,float* res,
	   bool test)
    {
        assert(bs==auto_dao::Batch_Size);
		for(int i=0;i<bs;i++)
		{
			int ad1=i*d1*h1*w1,ad2=i*d2*h2*w2;
			for(int j=0;j<d2;j++)
			{
				int adj=(j%d1)*h1*w1;
				for(int k=0;k<h2;k++)
				{
					int adk=(k%h1)*w1;
					for(int l=0;l<w2;l++)
					{
						res[ad2++]=a[ad1+adj+adk+(l%w1)];
					}
				}
			}
			if(test) break;
		}
    };
	res.dat->backward_f=[&](int bs,
	   int d1,int h1,int w1,float* a,float *da,
	   int d2,int h2,int w2,float* dres)
    {
        assert(bs==auto_dao::Batch_Size);
        memset(da,0,sizeof(float)*bs*d1*h1*w1);
		for(int i=0;i<bs;i++)
		{
			int ad1=i*d1*h1*w1,ad2=i*d2*h2*w2;
			for(int j=0;j<d2;j++)
			{
				int adj=(j%d1)*h1*w1;
				for(int k=0;k<h2;k++)
				{
					int adk=(k%h1)*w1;
					for(int l=0;l<w2;l++)
					{
						da[ad1+adj+adk+(l%w1)]+=dres[ad2++];
					}
				}
			}
		}
    };
    res.dat->forward();
    return res;
}

inline val3d operator+(val3d x,val3d y)
{
	assert(x.d==y.d&&x.h==y.h&&x.w==y.w);
	val3d res(x.d,x.h,x.w);
	res.dat->in1=x.dat;
	res.dat->in2=y.dat;
	x.dat->oud++;
	y.dat->oud++;
	res.dat->forward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,
	   int d2,int h2,int w2,float* a2,
	   int d3,int h3,int w3,float* res,
	   bool test)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1==d2&&d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) res[ad+j]=a1[ad+j]+a2[ad+j];
			if(test) break;
		}
    };
	res.dat->backward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,float *da1,
	   int d2,int h2,int w2,float* a2,float *da2,
	   int d3,int h3,int w3,float* dres)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1==d2&&d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) da1[ad+j]=dres[ad+j],da2[ad+j]=dres[ad+j];
		}
    };
    res.dat->forward();
    return res;
}

inline val3d operator-(val3d x,val3d y)
{
	assert(x.d==y.d&&x.h==y.h&&x.w==y.w);
	val3d res(x.d,x.h,x.w);
	res.dat->in1=x.dat;
	res.dat->in2=y.dat;
	x.dat->oud++;
	y.dat->oud++;
	res.dat->forward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,
	   int d2,int h2,int w2,float* a2,
	   int d3,int h3,int w3,float* res,
	   bool test)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1==d2&&d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) res[ad+j]=a1[ad+j]-a2[ad+j];
			if(test) break;
		}
    };
	res.dat->backward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,float *da1,
	   int d2,int h2,int w2,float* a2,float *da2,
	   int d3,int h3,int w3,float* dres)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1==d2&&d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) da1[ad+j]=dres[ad+j],da2[ad+j]=-dres[ad+j];
		}
    };
    res.dat->forward();
    return res;
}

inline val3d operator*(val3d x,val3d y)
{
	assert(x.d==y.d&&x.h==y.h&&x.w==y.w);
	val3d res(x.d,x.h,x.w);
	res.dat->in1=x.dat;
	res.dat->in2=y.dat;
	x.dat->oud++;
	y.dat->oud++;
	res.dat->forward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,
	   int d2,int h2,int w2,float* a2,
	   int d3,int h3,int w3,float* res,
	   bool test)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1==d2&&d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) res[ad+j]=a1[ad+j]*a2[ad+j];
			if(test) break;
		}
    };
	res.dat->backward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,float *da1,
	   int d2,int h2,int w2,float* a2,float *da2,
	   int d3,int h3,int w3,float* dres)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1==d2&&d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++)
			{
				da1[ad+j]=a2[ad+j]*dres[ad+j];
				da2[ad+j]=a1[ad+j]*dres[ad+j];
			}
		}
    };
    res.dat->forward();
    return res;
}

inline val3d operator/(val3d x,val3d y)
{
	assert(x.d==y.d&&x.h==y.h&&x.w==y.w);
	val3d res(x.d,x.h,x.w);
	res.dat->in1=x.dat;
	res.dat->in2=y.dat;
	x.dat->oud++;
	y.dat->oud++;
	res.dat->forward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,
	   int d2,int h2,int w2,float* a2,
	   int d3,int h3,int w3,float* res,
	   bool test)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1==d2&&d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) res[ad+j]=a1[ad+j]/a2[ad+j];
			if(test) break;
		}
    };
	res.dat->backward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,float *da1,
	   int d2,int h2,int w2,float* a2,float *da2,
	   int d3,int h3,int w3,float* dres)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1==d2&&d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad=i*d1*h1*w1;
			for(int j=0;j<d1*h1*w1;j++)
			{
				da1[ad+j]=dres[ad+j]/a2[ad+j];
				da2[ad+j]=-a1[ad+j]/(a2[ad+j]*a2[ad+j])*dres[ad+j];
			}
		}
    };
    res.dat->forward();
    return res;
}

inline val3d dcat(val3d x,val3d y)
{
	assert(x.h==y.h&&x.w==y.w);
	val3d res(x.d+y.d,x.h,x.w);
	res.dat->in1=x.dat;
	res.dat->in2=y.dat;
	x.dat->oud++;
	y.dat->oud++;
	res.dat->forward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,
	   int d2,int h2,int w2,float* a2,
	   int d3,int h3,int w3,float* res,
	   bool test)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1+d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad1=i*d1*h1*w1,ad2=i*d2*h1*w1;
			int ad=i*d3*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) res[ad+j]=a1[ad1+j];
			for(int j=0;j<d2*h1*w1;j++) res[ad+d1*h1*w1+j]=a2[ad2+j];
			if(test) break;
		}
    };
	res.dat->backward_f2=[&](int bs,
	   int d1,int h1,int w1,float* a1,float *da1,
	   int d2,int h2,int w2,float* a2,float *da2,
	   int d3,int h3,int w3,float* dres)
    {
        assert(bs==auto_dao::Batch_Size
	   	 	 &&d1+d2==d3
		 	 &&h1==h2&&h2==h3
			 &&w1==w2&&w2==w3);
		for(int i=0;i<bs;i++)
		{
			int ad1=i*d1*h1*w1,ad2=i*d2*h1*w1;
			int ad=i*d3*h1*w1;
			for(int j=0;j<d1*h1*w1;j++) da1[ad1+j]=dres[ad+j];
			for(int j=0;j<d2*h1*w1;j++) da2[ad2+j]=dres[ad+d1*h1*w1+j];
		}
    };
    res.dat->forward();
    return res;
}

inline float MSEloss(val3d x,float *realout)
{
	float res=0;
	int bs=auto_dao::test?1:auto_dao::Batch_Size;
	int n=bs*x.d*x.h*x.w;
	for(int i=0;i<n;i++)
	{
		float y=realout[i];
		res+=pow(x.a[i]-y,2)/n;
		x.da[i]=2*(x.a[i]-y)/n;
	}
	return res;
}

inline float BCEloss(val3d x,float *realout)
{
	float res=0;
	int bs=auto_dao::test?1:auto_dao::Batch_Size;
	int n=bs*x.d*x.h*x.w;
	for(int i=0;i<n;i++)
	{
		float val=x.a[i],y=realout[i];
		float eps=1e-7;
		if(val<eps||val>1-eps)
		{
			val=std::max(eps,std::min(val,1-eps));
			x.da[i]=0;
		}
		else x.da[i]=-(y*(1/val)-(1-y)*(1/(1-val)))/n;
		res+=-(y*log(val)+(1-y)*log(1-val))/n;
	}
	return res;
}
