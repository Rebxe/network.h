#pragma once

#include <cstring>

inline void img2col(
	int ind, int inh, int inw,
	int ch, int cw,
	int stx, int sty,
	int pdx, int pdy,
	float pdval,
	float *in, float *out)
{
	int cnt=0;
	for (int i = -pdx; i + ch <= inh + pdx; i += stx)
		for (int j = -pdy; j + cw <= inw + pdy; j += sty)
			for (int d = 0; d < ind; d++)
				for (int x = i; x < i + ch; x++)
					for (int y = j; y < j + cw; y++)
					{
						float &cur = out[cnt++];
						if(x>=0&&x<inh&&y>=0&&y<inw) cur = in[d * inh * inw + x * inw + y];
						else cur = pdval;
					}
}

inline void col2img( // simply add
	int ind, int inh, int inw,
	int ch, int cw,
	int stx, int sty,
	int pdx, int pdy,
	float *in, float *out)
{
	int cnt=0;
	for (int i = -pdx; i + ch <= inh + pdx; i += stx)
		for (int j = -pdy; j + cw <= inw + pdy; j += sty)
			for (int d = 0; d < ind; d++)
				for (int x = i; x < i + ch; x++)
					for (int y = j; y < j + cw; y++)
					{
						float cur = in[cnt++];
						if(x>=0&&x<inh&&y>=0&&y<inw) out[d * inh * inw + x * inw + y] += cur;
					}
}

#ifdef ENABLE_GPU
	#define _SILENCE_AMP_DEPRECATION_WARNINGS 0
	#include <amp.h>
	#include <amp_math.h>
	/*
		A: N * M
		B: M * K
		Res: K * N
	*/
    void Matrix_Mul(int bs, int N, int M, int K, float* A, float* B, float* Res)
    {
        concurrency::array_view<float, 2> a(N, M, A);
        concurrency::array_view<float, 2> b(M, K, B);
        concurrency::array_view<float, 2> res(K, N, Res);
        res.discard_data();

        concurrency::parallel_for_each(res.extent,
            [=](concurrency::index<2> idx) restrict(amp) {
                int i = idx[1], k = idx[0];
                res[idx] = 0;
                for (int j = 0; j < M; j++) res[idx] += a(i, j) * b(j, k);
            });

        res.synchronize();
    }
    void Matrix_Mul_Back_B(int N, int M, int K, float* A, float* B, float* Res) // simply add
    {
        concurrency::array_view<float, 2> a(N, M, A);
        concurrency::array_view<float, 2> b(M, K, B);
        concurrency::array_view<float, 2> res(K, N, Res);

        concurrency::parallel_for_each(b.extent,
            [=](concurrency::index<2> idx) restrict(amp) {
                int i = idx[0], k = idx[1];
                for (int j = 0; j < N; j++) b[idx] += a(j, i) * res(k, j);
            });

        b.synchronize();
    }
    void Matrix_Mul_Back_A(int N, int M, int K, float* A, float* B, float* Res)
    {
        concurrency::array_view<float, 2> a(N, M, A);
        concurrency::array_view<float, 2> b(M, K, B);
        concurrency::array_view<float, 2> res(K, N, Res);
        a.discard_data();

        concurrency::parallel_for_each(a.extent,
            [=](concurrency::index<2> idx) restrict(amp) {
                int i = idx[0], k = idx[1];
                a[idx] = 0;
                for (int j = 0; j < K; j++) a[idx] += res(j, i) * b(k, j);
            });

        a.synchronize();
    }
#else
	/*
		A: N * M
		B: M * K
		res: K * N
	*/
	#ifdef __AVX__
  		#include <immintrin.h>
	    void Matrix_Mul(int N, int M, int K, float* A, float* B, float* Res)
	    {
	        float *packb[4];
			for(int i=0;i<4;i++) packb[i]=new float[M];
			float tmp[8];
	        for(int j=0;j+3<K;j+=4)
	        {
	        	for(int k=0;k<M;k++)
	       		{
	       			packb[0][k]=B[k*K+(j+0)];
	       			packb[1][k]=B[k*K+(j+1)];
	       			packb[2][k]=B[k*K+(j+2)];
	       			packb[3][k]=B[k*K+(j+3)];
	       	    }
	        	for(int i=0;i+3<N;i+=4)
	        	{
	        		{
		       			register __m256 r00,r01,r02,r03;
		       			register __m256 r10,r11,r12,r13;
		       			register __m256 r20,r21,r22,r23;
		       			register __m256 r30,r31,r32,r33;
		       			r00=r01=r02=r03=
		       			r10=r11=r12=r13=
		       			r20=r21=r22=r23=
		       			r30=r31=r32=r33=_mm256_set1_ps(0);
		       			float *pa0=A+(i+0)*M;
						float *pa1=A+(i+1)*M;
						float *pa2=A+(i+2)*M;
						float *pa3=A+(i+3)*M;
		       			float *pb0=packb[0];
						float *pb1=packb[1];
						float *pb2=packb[2];
						float *pb3=packb[3];
		           		for(int k=0;k+7<M;k+=8)
		           		{
		           			__m256 a0=_mm256_loadu_ps(pa0);
		           			__m256 a1=_mm256_loadu_ps(pa1);
		           			__m256 a2=_mm256_loadu_ps(pa2);
		           			__m256 a3=_mm256_loadu_ps(pa3);
		           			
		           			__m256 b0=_mm256_loadu_ps(pb0);
		           			__m256 b1=_mm256_loadu_ps(pb1);
		           			__m256 b2=_mm256_loadu_ps(pb2);
		           			__m256 b3=_mm256_loadu_ps(pb3);
		           			
		           			r00=_mm256_fmadd_ps(b0,a0,r00);
		           			r01=_mm256_fmadd_ps(b0,a1,r01);
		           			r02=_mm256_fmadd_ps(b0,a2,r02);
		           			r03=_mm256_fmadd_ps(b0,a3,r03);
		           			
		           			r10=_mm256_fmadd_ps(b1,a0,r10);
		           			r11=_mm256_fmadd_ps(b1,a1,r11);
		           			r12=_mm256_fmadd_ps(b1,a2,r12);
		           			r13=_mm256_fmadd_ps(b1,a3,r13);
		           			
		           			r20=_mm256_fmadd_ps(b2,a0,r20);
		           			r21=_mm256_fmadd_ps(b2,a1,r21);
		           			r22=_mm256_fmadd_ps(b2,a2,r22);
		           			r23=_mm256_fmadd_ps(b2,a3,r23);
		           			
		           			r30=_mm256_fmadd_ps(b3,a0,r30);
		           			r31=_mm256_fmadd_ps(b3,a1,r31);
		           			r32=_mm256_fmadd_ps(b3,a2,r32);
		           			r33=_mm256_fmadd_ps(b3,a3,r33);
		           	        
		           	        pa0+=8,pa1+=8,pa2+=8,pa3+=8; 
		           	        pb0+=8,pb1+=8,pb2+=8,pb3+=8;
		           	    }
		           	    _mm256_storeu_ps(tmp,r00);
		       	        Res[(j+0)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r01);
		       	        Res[(j+0)*N+(i+1)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r02);
		       	        Res[(j+0)*N+(i+2)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r03);
		       	        Res[(j+0)*N+(i+3)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	        
		           	    _mm256_storeu_ps(tmp,r10);
		       	        Res[(j+1)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r11);
		       	        Res[(j+1)*N+(i+1)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r12);
		       	        Res[(j+1)*N+(i+2)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r13);
		       	        Res[(j+1)*N+(i+3)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	        
		           	    _mm256_storeu_ps(tmp,r20);
		       	        Res[(j+2)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r21);
		       	        Res[(j+2)*N+(i+1)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r22);
		       	        Res[(j+2)*N+(i+2)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r23);
		       	        Res[(j+2)*N+(i+3)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	        
		           	    _mm256_storeu_ps(tmp,r30);
		       	        Res[(j+3)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r31);
		       	        Res[(j+3)*N+(i+1)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r32);
		       	        Res[(j+3)*N+(i+2)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r33);
		       	        Res[(j+3)*N+(i+3)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	    }
	       			register float r00=0,r01=0,r02=0,r03=0;
	       			register float r10=0,r11=0,r12=0,r13=0;
	       			register float r20=0,r21=0,r22=0,r23=0;
	       			register float r30=0,r31=0,r32=0,r33=0;
	       			float *a0=A+(i+0)*M+M-M%8;
					float *a1=A+(i+1)*M+M-M%8;
					float *a2=A+(i+2)*M+M-M%8;
					float *a3=A+(i+3)*M+M-M%8;
	       			float *b0=packb[0]+M-M%8;
					float *b1=packb[1]+M-M%8;
					float *b2=packb[2]+M-M%8;
					float *b3=packb[3]+M-M%8;
	           		for(int k=M-M%8;k<M;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r01+=(*b0)*(*a1);
	           	        r02+=(*b0)*(*a2);
	           	        r03+=(*b0)*(*a3);
	           	        
	           	        r10+=(*b1)*(*a0);
	           	        r11+=(*b1)*(*a1);
	           	        r12+=(*b1)*(*a2);
	           	        r13+=(*b1)*(*a3);
	           	        
	           	        r20+=(*b2)*(*a0);
	           	        r21+=(*b2)*(*a1);
	           	        r22+=(*b2)*(*a2);
	           	        r23+=(*b2)*(*a3);
	           	        
	           	        r30+=(*b3)*(*a0);
	           	        r31+=(*b3)*(*a1);
	           	        r32+=(*b3)*(*a2);
	           	        r33+=(*b3)*(*a3);
	           	        
	           	        a0++,a1++,a2++,a3++;
	           	        b0++,b1++,b2++,b3++;
	           	    }
	       	        Res[(j+0)*N+(i+0)]+=r00;
	       	        Res[(j+0)*N+(i+1)]+=r01;
	       	        Res[(j+0)*N+(i+2)]+=r02;
	       	        Res[(j+0)*N+(i+3)]+=r03;
	       	        
	       	        Res[(j+1)*N+(i+0)]+=r10;
	       	        Res[(j+1)*N+(i+1)]+=r11;
	       	        Res[(j+1)*N+(i+2)]+=r12;
	       	        Res[(j+1)*N+(i+3)]+=r13;
	       	        
	       	        Res[(j+2)*N+(i+0)]+=r20;
	       	        Res[(j+2)*N+(i+1)]+=r21;
	       	        Res[(j+2)*N+(i+2)]+=r22;
	       	        Res[(j+2)*N+(i+3)]+=r23;
	       	        
	       	        Res[(j+3)*N+(i+0)]+=r30;
	       	        Res[(j+3)*N+(i+1)]+=r31;
	       	        Res[(j+3)*N+(i+2)]+=r32;
	       	        Res[(j+3)*N+(i+3)]+=r33;
				}
				for(int i=N-N%4;i<N;i++)
				{
	        		{
		       			register __m256 r00,r10,r20,r30;
		       			r00=r10=r20=r30=_mm256_set1_ps(0);
		       			float *pa0=A+(i+0)*M;
		       			float *pb0=packb[0];
						float *pb1=packb[1];
						float *pb2=packb[2];
						float *pb3=packb[3];
		           		for(int k=0;k+7<M;k+=8)
		           		{
		           			__m256 a0=_mm256_loadu_ps(pa0);
		           			
		           			__m256 b0=_mm256_loadu_ps(pb0);
		           			__m256 b1=_mm256_loadu_ps(pb1);
		           			__m256 b2=_mm256_loadu_ps(pb2);
		           			__m256 b3=_mm256_loadu_ps(pb3);
		           			
		           			r00=_mm256_fmadd_ps(b0,a0,r00);
		           			r10=_mm256_fmadd_ps(b1,a0,r10);
		           			r20=_mm256_fmadd_ps(b2,a0,r20);
		           			r30=_mm256_fmadd_ps(b3,a0,r30);
		           	        
		           	        pa0+=8; 
		           	        pb0+=8,pb1+=8,pb2+=8,pb3+=8;
		           	    }
		           	    _mm256_storeu_ps(tmp,r00);
		       	        Res[(j+0)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r10);
		       	        Res[(j+1)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r20);
		       	        Res[(j+2)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r30);
		       	        Res[(j+3)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	    }
	       			register float r00=0,r10=0,r20=0,r30=0;
	       			float *a0=A+(i+0)*M+M-M%8;
	       			float *b0=packb[0]+M-M%8;
					float *b1=packb[1]+M-M%8;
					float *b2=packb[2]+M-M%8;
					float *b3=packb[3]+M-M%8;
	           		for(int k=M-M%8;k<M;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r10+=(*b1)*(*a0);
	           	        r20+=(*b2)*(*a0);
	           	        r30+=(*b3)*(*a0);
	           	        
	           	        a0++;
	           	        b0++,b1++,b2++,b3++;
	           	    }
	       	        Res[(j+0)*N+(i+0)]+=r00;
	       	        Res[(j+1)*N+(i+0)]+=r10;
	       	        Res[(j+2)*N+(i+0)]+=r20;
	       	        Res[(j+3)*N+(i+0)]+=r30;
	           	}
			}
	        for(int j=K-K%4;j<K;j++)
	        {
	       		for(int k=0;k<M;k++) packb[0][k]=B[k*K+(j+0)];
	        	for(int i=0;i+3<N;i+=4)
	        	{
	        		{
		       			register __m256 r00,r01,r02,r03;
		       			r00=r01=r02=r03=_mm256_set1_ps(0);
		       			float *pa0=A+(i+0)*M;
						float *pa1=A+(i+1)*M;
						float *pa2=A+(i+2)*M;
						float *pa3=A+(i+3)*M;
		       			float *pb0=packb[0];
		           		for(int k=0;k+7<M;k+=8)
		           		{
		           			__m256 a0=_mm256_loadu_ps(pa0);
		           			__m256 a1=_mm256_loadu_ps(pa1);
		           			__m256 a2=_mm256_loadu_ps(pa2);
		           			__m256 a3=_mm256_loadu_ps(pa3);
		           			
		           			__m256 b0=_mm256_loadu_ps(pb0);
		           			
		           			r00=_mm256_fmadd_ps(b0,a0,r00);
		           			r01=_mm256_fmadd_ps(b0,a1,r01);
		           			r02=_mm256_fmadd_ps(b0,a2,r02);
		           			r03=_mm256_fmadd_ps(b0,a3,r03);
		           	        
		           	        pa0+=8,pa1+=8,pa2+=8,pa3+=8; 
		           	        pb0+=8;
		           	    }
		           	    _mm256_storeu_ps(tmp,r00);
		       	        Res[(j+0)*N+(i+0)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r01);
		       	        Res[(j+0)*N+(i+1)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r02);
		       	        Res[(j+0)*N+(i+2)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r03);
		       	        Res[(j+0)*N+(i+3)]=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	    }
	       			register float r00=0,r01=0,r02=0,r03=0;
	       			float *a0=A+(i+0)*M+M-M%8;
					float *a1=A+(i+1)*M+M-M%8;
					float *a2=A+(i+2)*M+M-M%8;
					float *a3=A+(i+3)*M+M-M%8;
	       			float *b0=packb[0]+M-M%8;
	           		for(int k=M-M%8;k<M;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r01+=(*b0)*(*a1);
	           	        r02+=(*b0)*(*a2);
	           	        r03+=(*b0)*(*a3);
	           	        
	           	        a0++,a1++,a2++,a3++;
	           	        b0++;
	           	    }
	       	        Res[(j+0)*N+(i+0)]+=r00;
	       	        Res[(j+0)*N+(i+1)]+=r01;
	       	        Res[(j+0)*N+(i+2)]+=r02;
	       	        Res[(j+0)*N+(i+3)]+=r03;
				}
			}
	        for(int j=K-K%4;j<K;j++)
				for(int i=N-N%4;i<N;i++)
				{
					Res[j*N+i]=0;
	           		for(int k=0;k<M;k++)
	           	        Res[j*N+i]+=A[i*M+k]*B[k*K+j];
	           	}
	        for(int i=0;i<4;i++) delete[] packb[i];
	    }
	    void Matrix_Mul_Back_B(int N, int M, int K, float* A, float* B, float* Res) // simply add
	    {
	        float *packb[4];
			for(int i=0;i<4;i++) packb[i]=new float[N];
			float tmp[8];
	        for(int j=0;j+3<M;j+=4)
	        {
	       		for(int k=0;k<N;k++)
	       		{
	       			packb[0][k]=A[k*M+(j+0)];
	       			packb[1][k]=A[k*M+(j+1)];
	       			packb[2][k]=A[k*M+(j+2)];
	       			packb[3][k]=A[k*M+(j+3)];
	       	    }
	        	for(int i=0;i+3<K;i+=4)
	        	{
	        		{
		       			register __m256 r00,r01,r02,r03;
		       			register __m256 r10,r11,r12,r13;
		       			register __m256 r20,r21,r22,r23;
		       			register __m256 r30,r31,r32,r33;
		       			r00=r01=r02=r03=
		       			r10=r11=r12=r13=
		       			r20=r21=r22=r23=
		       			r30=r31=r32=r33=_mm256_set1_ps(0);
	       				float *pa0=Res+(i+0)*N;
	       				float *pa1=Res+(i+1)*N;
	       				float *pa2=Res+(i+2)*N;
	       				float *pa3=Res+(i+3)*N;
		       			float *pb0=packb[0];
						float *pb1=packb[1];
						float *pb2=packb[2];
						float *pb3=packb[3];
		           		for(int k=0;k+7<N;k+=8)
		           		{
		           			__m256 a0=_mm256_loadu_ps(pa0);
		           			__m256 a1=_mm256_loadu_ps(pa1);
		           			__m256 a2=_mm256_loadu_ps(pa2);
		           			__m256 a3=_mm256_loadu_ps(pa3);
		           			
		           			__m256 b0=_mm256_loadu_ps(pb0);
		           			__m256 b1=_mm256_loadu_ps(pb1);
		           			__m256 b2=_mm256_loadu_ps(pb2);
		           			__m256 b3=_mm256_loadu_ps(pb3);
		           			
		           			r00=_mm256_fmadd_ps(b0,a0,r00);
		           			r01=_mm256_fmadd_ps(b0,a1,r01);
		           			r02=_mm256_fmadd_ps(b0,a2,r02);
		           			r03=_mm256_fmadd_ps(b0,a3,r03);
		           			
		           			r10=_mm256_fmadd_ps(b1,a0,r10);
		           			r11=_mm256_fmadd_ps(b1,a1,r11);
		           			r12=_mm256_fmadd_ps(b1,a2,r12);
		           			r13=_mm256_fmadd_ps(b1,a3,r13);
		           			
		           			r20=_mm256_fmadd_ps(b2,a0,r20);
		           			r21=_mm256_fmadd_ps(b2,a1,r21);
		           			r22=_mm256_fmadd_ps(b2,a2,r22);
		           			r23=_mm256_fmadd_ps(b2,a3,r23);
		           			
		           			r30=_mm256_fmadd_ps(b3,a0,r30);
		           			r31=_mm256_fmadd_ps(b3,a1,r31);
		           			r32=_mm256_fmadd_ps(b3,a2,r32);
		           			r33=_mm256_fmadd_ps(b3,a3,r33);
		           	        
		           	        pa0+=8,pa1+=8,pa2+=8,pa3+=8; 
		           	        pb0+=8,pb1+=8,pb2+=8,pb3+=8;
		           	    }
		           	    _mm256_storeu_ps(tmp,r00);
		       	        B[(j+0)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r01);
		       	        B[(j+0)*K+(i+1)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r02);
		       	        B[(j+0)*K+(i+2)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r03);
		       	        B[(j+0)*K+(i+3)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	        
		           	    _mm256_storeu_ps(tmp,r10);
		       	        B[(j+1)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r11);
		       	        B[(j+1)*K+(i+1)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r12);
		       	        B[(j+1)*K+(i+2)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r13);
		       	        B[(j+1)*K+(i+3)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	        
		           	    _mm256_storeu_ps(tmp,r20);
		       	        B[(j+2)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r21);
		       	        B[(j+2)*K+(i+1)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r22);
		       	        B[(j+2)*K+(i+2)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r23);
		       	        B[(j+2)*K+(i+3)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	        
		           	    _mm256_storeu_ps(tmp,r30);
		       	        B[(j+3)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r31);
		       	        B[(j+3)*K+(i+1)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r32);
		       	        B[(j+3)*K+(i+2)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r33);
		       	        B[(j+3)*K+(i+3)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	    }
	       			register float r00=0,r01=0,r02=0,r03=0;
	       			register float r10=0,r11=0,r12=0,r13=0;
	       			register float r20=0,r21=0,r22=0,r23=0;
	       			register float r30=0,r31=0,r32=0,r33=0;
	       			float *a0=Res+(i+0)*N+N-N%8;
					float *a1=Res+(i+1)*N+N-N%8;
					float *a2=Res+(i+2)*N+N-N%8;
					float *a3=Res+(i+3)*N+N-N%8;
	       			float *b0=packb[0]+N-N%8;
					float *b1=packb[1]+N-N%8;
					float *b2=packb[2]+N-N%8;
					float *b3=packb[3]+N-N%8;
	           		for(int k=N-N%8;k<N;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r01+=(*b0)*(*a1);
	           	        r02+=(*b0)*(*a2);
	           	        r03+=(*b0)*(*a3);
	           	        
	           	        r10+=(*b1)*(*a0);
	           	        r11+=(*b1)*(*a1);
	           	        r12+=(*b1)*(*a2);
	           	        r13+=(*b1)*(*a3);
	           	        
	           	        r20+=(*b2)*(*a0);
	           	        r21+=(*b2)*(*a1);
	           	        r22+=(*b2)*(*a2);
	           	        r23+=(*b2)*(*a3);
	           	        
	           	        r30+=(*b3)*(*a0);
	           	        r31+=(*b3)*(*a1);
	           	        r32+=(*b3)*(*a2);
	           	        r33+=(*b3)*(*a3);
	           	        
	           	        a0++,a1++,a2++,a3++;
	           	        b0++,b1++,b2++,b3++;
	           	    }
	       	        B[(j+0)*K+(i+0)]+=r00;
	       	        B[(j+0)*K+(i+1)]+=r01;
	       	        B[(j+0)*K+(i+2)]+=r02;
	       	        B[(j+0)*K+(i+3)]+=r03;
	       	        
	       	        B[(j+1)*K+(i+0)]+=r10;
	       	        B[(j+1)*K+(i+1)]+=r11;
	       	        B[(j+1)*K+(i+2)]+=r12;
	       	        B[(j+1)*K+(i+3)]+=r13;
	       	        
	       	        B[(j+2)*K+(i+0)]+=r20;
	       	        B[(j+2)*K+(i+1)]+=r21;
	       	        B[(j+2)*K+(i+2)]+=r22;
	       	        B[(j+2)*K+(i+3)]+=r23;
	       	        
	       	        B[(j+3)*K+(i+0)]+=r30;
	       	        B[(j+3)*K+(i+1)]+=r31;
	       	        B[(j+3)*K+(i+2)]+=r32;
	       	        B[(j+3)*K+(i+3)]+=r33;
				}
	        	for(int i=K-K%4;i<K;i++)
	        	{
	        		{
		       			register __m256 r00,r10,r20,r30;
		       			r00=r10=r20=r30=_mm256_set1_ps(0);
	       				float *pa0=Res+(i+0)*N;
		       			float *pb0=packb[0];
						float *pb1=packb[1];
						float *pb2=packb[2];
						float *pb3=packb[3];
		           		for(int k=0;k+7<N;k+=8)
		           		{
		           			__m256 a0=_mm256_loadu_ps(pa0);
		           			
		           			__m256 b0=_mm256_loadu_ps(pb0);
		           			__m256 b1=_mm256_loadu_ps(pb1);
		           			__m256 b2=_mm256_loadu_ps(pb2);
		           			__m256 b3=_mm256_loadu_ps(pb3);
		           			
		           			r00=_mm256_fmadd_ps(b0,a0,r00);
		           			r10=_mm256_fmadd_ps(b1,a0,r10);
		           			r20=_mm256_fmadd_ps(b2,a0,r20);
		           			r30=_mm256_fmadd_ps(b3,a0,r30);
		           	        
		           	        pa0+=8; 
		           	        pb0+=8,pb1+=8,pb2+=8,pb3+=8;
		           	    }
		           	    _mm256_storeu_ps(tmp,r00);
		       	        B[(j+0)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r10);
		       	        B[(j+1)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r20);
		       	        B[(j+2)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r30);
		       	        B[(j+3)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	    }
	       			register float r00=0,r10=0,r20=0,r30=0;
	       			float *a0=Res+(i+0)*N+N-N%8;
	       			float *b0=packb[0]+N-N%8;
					float *b1=packb[1]+N-N%8;
					float *b2=packb[2]+N-N%8;
					float *b3=packb[3]+N-N%8;
	           		for(int k=N-N%8;k<N;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r10+=(*b1)*(*a0);
	           	        r20+=(*b2)*(*a0);
	           	        r30+=(*b3)*(*a0);
	           	        
	           	        a0++;
	           	        b0++,b1++,b2++,b3++;
	           	    }
	       	        B[(j+0)*K+(i+0)]+=r00;
	       	        B[(j+1)*K+(i+0)]+=r10;
	       	        B[(j+2)*K+(i+0)]+=r20;
	       	        B[(j+3)*K+(i+0)]+=r30;
				}
			}
	        for(int j=M-M%4;j<M;j++)
	        {
	       		for(int k=0;k<N;k++) packb[0][k]=A[k*M+(j+0)];
	        	for(int i=0;i+3<K;i+=4)
	        	{
	        		{
		       			register __m256 r00,r01,r02,r03;
		       			r00=r01=r02=r03=_mm256_set1_ps(0);
	       				float *pa0=Res+(i+0)*N;
	       				float *pa1=Res+(i+1)*N;
	       				float *pa2=Res+(i+2)*N;
	       				float *pa3=Res+(i+3)*N;
		       			float *pb0=packb[0];
		           		for(int k=0;k+7<N;k+=8)
		           		{
		           			__m256 a0=_mm256_loadu_ps(pa0);
		           			__m256 a1=_mm256_loadu_ps(pa1);
		           			__m256 a2=_mm256_loadu_ps(pa2);
		           			__m256 a3=_mm256_loadu_ps(pa3);
		           			
		           			__m256 b0=_mm256_loadu_ps(pb0);
		           			
		           			r00=_mm256_fmadd_ps(b0,a0,r00);
		           			r01=_mm256_fmadd_ps(b0,a1,r01);
		           			r02=_mm256_fmadd_ps(b0,a2,r02);
		           			r03=_mm256_fmadd_ps(b0,a3,r03);
		           	        
		           	        pa0+=8,pa1+=8,pa2+=8,pa3+=8; 
		           	        pb0+=8;
		           	    }
		           	    _mm256_storeu_ps(tmp,r00);
		       	        B[(j+0)*K+(i+0)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r01);
		       	        B[(j+0)*K+(i+1)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r02);
		       	        B[(j+0)*K+(i+2)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		           	    _mm256_storeu_ps(tmp,r03);
		       	        B[(j+0)*K+(i+3)]+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
		       	    }
	       			register float r00=0,r01=0,r02=0,r03=0;
	       			float *a0=Res+(i+0)*N+N-N%8;
					float *a1=Res+(i+1)*N+N-N%8;
					float *a2=Res+(i+2)*N+N-N%8;
					float *a3=Res+(i+3)*N+N-N%8;
	       			float *b0=packb[0]+N-N%8;
	           		for(int k=N-N%8;k<N;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r01+=(*b0)*(*a1);
	           	        r02+=(*b0)*(*a2);
	           	        r03+=(*b0)*(*a3);
	           	        
	           	        a0++,a1++,a2++,a3++;
	           	        b0++;
	           	    }
	       	        B[(j+0)*K+(i+0)]+=r00;
	       	        B[(j+0)*K+(i+1)]+=r01;
	       	        B[(j+0)*K+(i+2)]+=r02;
	       	        B[(j+0)*K+(i+3)]+=r03;
				}
			}
	        for(int j=M-M%4;j<M;j++)
	            for(int i=K-K%4;i<K;i++)
			        for(int k=0;k<N;k++)
						B[j*K+i]+=A[k*M+j]*Res[i*N+k];
	        for(int i=0;i<4;i++) delete[] packb[i];
	    }
	#else
		#ifndef DISABLE_AVX
			#error use '-mavx2 -mfma' to enable AVX256, define DISABLE_AVX to ignore
		#endif
	    void Matrix_Mul(int N, int M, int K, float* A, float* B, float* Res)
	    {
	        float *packb[4];
			for(int i=0;i<4;i++) packb[i]=new float[M];
	        for(int j=0;j+3<K;j+=4)
	        {
	       		for(int k=0;k<M;k++)
	       		{
	       			packb[0][k]=B[k*K+(j+0)];
	       			packb[1][k]=B[k*K+(j+1)];
	       			packb[2][k]=B[k*K+(j+2)];
	       			packb[3][k]=B[k*K+(j+3)];
	       	    }
	        	for(int i=0;i+3<N;i+=4)
	        	{
	       			register float r00=0,r01=0,r02=0,r03=0;
	       			register float r10=0,r11=0,r12=0,r13=0;
	       			register float r20=0,r21=0,r22=0,r23=0;
	       			register float r30=0,r31=0,r32=0,r33=0;
	       			float *a0=A+(i+0)*M;
					float *a1=A+(i+1)*M;
					float *a2=A+(i+2)*M;
					float *a3=A+(i+3)*M;
	       			float *b0=packb[0];
					float *b1=packb[1];
					float *b2=packb[2];
					float *b3=packb[3];
	           		for(int k=0;k<M;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r01+=(*b0)*(*a1);
	           	        r02+=(*b0)*(*a2);
	           	        r03+=(*b0)*(*a3);
	           	        
	           	        r10+=(*b1)*(*a0);
	           	        r11+=(*b1)*(*a1);
	           	        r12+=(*b1)*(*a2);
	           	        r13+=(*b1)*(*a3);
	           	        
	           	        r20+=(*b2)*(*a0);
	           	        r21+=(*b2)*(*a1);
	           	        r22+=(*b2)*(*a2);
	           	        r23+=(*b2)*(*a3);
	           	        
	           	        r30+=(*b3)*(*a0);
	           	        r31+=(*b3)*(*a1);
	           	        r32+=(*b3)*(*a2);
	           	        r33+=(*b3)*(*a3);
	           	        
	           	        a0++,a1++,a2++,a3++;
	           	        b0++,b1++,b2++,b3++;
	           	    }
	       	        Res[(j+0)*N+(i+0)]=r00;
	       	        Res[(j+0)*N+(i+1)]=r01;
	       	        Res[(j+0)*N+(i+2)]=r02;
	       	        Res[(j+0)*N+(i+3)]=r03;
	       	        
	       	        Res[(j+1)*N+(i+0)]=r10;
	       	        Res[(j+1)*N+(i+1)]=r11;
	       	        Res[(j+1)*N+(i+2)]=r12;
	       	        Res[(j+1)*N+(i+3)]=r13;
	       	        
	       	        Res[(j+2)*N+(i+0)]=r20;
	       	        Res[(j+2)*N+(i+1)]=r21;
	       	        Res[(j+2)*N+(i+2)]=r22;
	       	        Res[(j+2)*N+(i+3)]=r23;
	       	        
	       	        Res[(j+3)*N+(i+0)]=r30;
	       	        Res[(j+3)*N+(i+1)]=r31;
	       	        Res[(j+3)*N+(i+2)]=r32;
	       	        Res[(j+3)*N+(i+3)]=r33;
				}
				for(int i=N-N%4;i<N;i++)
				{
	       			register float r00=0,r10=0,r20=0,r30=0;
	       			float *a0=A+(i+0)*M;
	       			float *b0=packb[0];
					float *b1=packb[1];
					float *b2=packb[2];
					float *b3=packb[3];
	           		for(int k=0;k<M;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r10+=(*b1)*(*a0);
	           	        r20+=(*b2)*(*a0);
	           	        r30+=(*b3)*(*a0);
	           	        
	           	        a0++;
	           	        b0++,b1++,b2++,b3++;
	           	    }
	       	        Res[(j+0)*N+(i+0)]=r00;
	       	        Res[(j+1)*N+(i+0)]=r10;
	       	        Res[(j+2)*N+(i+0)]=r20;
	       	        Res[(j+3)*N+(i+0)]=r30;
	           	}
			}
	        for(int j=K-K%4;j<K;j++)
	        {
	       		for(int k=0;k<M;k++) packb[0][k]=B[k*K+(j+0)];
	        	for(int i=0;i+3<N;i+=4)
	        	{
	       			register float r00=0,r01=0,r02=0,r03=0;
	       			float *a0=A+(i+0)*M;
					float *a1=A+(i+1)*M;
					float *a2=A+(i+2)*M;
					float *a3=A+(i+3)*M;
	       			float *b0=packb[0];
	           		for(int k=0;k<M;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r01+=(*b0)*(*a1);
	           	        r02+=(*b0)*(*a2);
	           	        r03+=(*b0)*(*a3);
	           	        
	           	        a0++,a1++,a2++,a3++;
	           	        b0++;
	           	    }
	       	        Res[(j+0)*N+(i+0)]=r00;
	       	        Res[(j+0)*N+(i+1)]=r01;
	       	        Res[(j+0)*N+(i+2)]=r02;
	       	        Res[(j+0)*N+(i+3)]=r03;
				}
			}
	        for(int j=K-K%4;j<K;j++)
				for(int i=N-N%4;i<N;i++)
				{
					Res[j*N+i]=0;
	           		for(int k=0;k<M;k++)
	           	        Res[j*N+i]+=A[i*M+k]*B[k*K+j];
	           	}
	        for(int i=0;i<4;i++) delete[] packb[i];
	    }
	    void Matrix_Mul_Back_B(int N, int M, int K, float* A, float* B, float* Res) // simply add
	    {
	        float *packb[4];
			for(int i=0;i<4;i++) packb[i]=new float[N];
	        for(int j=0;j+3<M;j+=4)
	        {
	       		for(int k=0;k<N;k++)
	       		{
	       			packb[0][k]=A[k*M+(j+0)];
	       			packb[1][k]=A[k*M+(j+1)];
	       			packb[2][k]=A[k*M+(j+2)];
	       			packb[3][k]=A[k*M+(j+3)];
	       	    }
	        	for(int i=0;i+3<K;i+=4)
	        	{
	       			register float r00=0,r01=0,r02=0,r03=0;
	       			register float r10=0,r11=0,r12=0,r13=0;
	       			register float r20=0,r21=0,r22=0,r23=0;
	       			register float r30=0,r31=0,r32=0,r33=0;
	       			float *a0=Res+(i+0)*N;
					float *a1=Res+(i+1)*N;
					float *a2=Res+(i+2)*N;
					float *a3=Res+(i+3)*N;
	       			float *b0=packb[0];
					float *b1=packb[1];
					float *b2=packb[2];
					float *b3=packb[3];
	           		for(int k=0;k<N;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r01+=(*b0)*(*a1);
	           	        r02+=(*b0)*(*a2);
	           	        r03+=(*b0)*(*a3);
	           	        
	           	        r10+=(*b1)*(*a0);
	           	        r11+=(*b1)*(*a1);
	           	        r12+=(*b1)*(*a2);
	           	        r13+=(*b1)*(*a3);
	           	        
	           	        r20+=(*b2)*(*a0);
	           	        r21+=(*b2)*(*a1);
	           	        r22+=(*b2)*(*a2);
	           	        r23+=(*b2)*(*a3);
	           	        
	           	        r30+=(*b3)*(*a0);
	           	        r31+=(*b3)*(*a1);
	           	        r32+=(*b3)*(*a2);
	           	        r33+=(*b3)*(*a3);
	           	        
	           	        a0++,a1++,a2++,a3++;
	           	        b0++,b1++,b2++,b3++;
	           	    }
	       	        B[(j+0)*K+(i+0)]+=r00;
	       	        B[(j+0)*K+(i+1)]+=r01;
	       	        B[(j+0)*K+(i+2)]+=r02;
	       	        B[(j+0)*K+(i+3)]+=r03;
	       	        
	       	        B[(j+1)*K+(i+0)]+=r10;
	       	        B[(j+1)*K+(i+1)]+=r11;
	       	        B[(j+1)*K+(i+2)]+=r12;
	       	        B[(j+1)*K+(i+3)]+=r13;
	       	        
	       	        B[(j+2)*K+(i+0)]+=r20;
	       	        B[(j+2)*K+(i+1)]+=r21;
	       	        B[(j+2)*K+(i+2)]+=r22;
	       	        B[(j+2)*K+(i+3)]+=r23;
	       	        
	       	        B[(j+3)*K+(i+0)]+=r30;
	       	        B[(j+3)*K+(i+1)]+=r31;
	       	        B[(j+3)*K+(i+2)]+=r32;
	       	        B[(j+3)*K+(i+3)]+=r33;
				}
	        	for(int i=K-K%4;i<K;i++)
	        	{
	       			register float r00=0,r10=0,r20=0,r30=0;
	       			float *a0=Res+(i+0)*N;
	       			float *b0=packb[0];
					float *b1=packb[1];
					float *b2=packb[2];
					float *b3=packb[3];
	           		for(int k=0;k<N;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r10+=(*b1)*(*a0);
	           	        r20+=(*b2)*(*a0);
	           	        r30+=(*b3)*(*a0);
	           	        
	           	        a0++;
	           	        b0++,b1++,b2++,b3++;
	           	    }
	       	        B[(j+0)*K+(i+0)]+=r00;
	       	        B[(j+1)*K+(i+0)]+=r10;
	       	        B[(j+2)*K+(i+0)]+=r20;
	       	        B[(j+3)*K+(i+0)]+=r30;
				}
			}
	        for(int j=M-M%4;j<M;j++)
	        {
	       		for(int k=0;k<N;k++) packb[0][k]=A[k*M+(j+0)];
	        	for(int i=0;i+3<K;i+=4)
	        	{
	       			register float r00=0,r01=0,r02=0,r03=0;
	       			float *a0=Res+(i+0)*N;
					float *a1=Res+(i+1)*N;
					float *a2=Res+(i+2)*N;
					float *a3=Res+(i+3)*N;
	       			float *b0=packb[0];
	           		for(int k=0;k<N;k++)
	           		{
	           	        r00+=(*b0)*(*a0);
	           	        r01+=(*b0)*(*a1);
	           	        r02+=(*b0)*(*a2);
	           	        r03+=(*b0)*(*a3);
	           	        
	           	        a0++,a1++,a2++,a3++;
	           	        b0++;
	           	    }
	       	        B[(j+0)*K+(i+0)]+=r00;
	       	        B[(j+0)*K+(i+1)]+=r01;
	       	        B[(j+0)*K+(i+2)]+=r02;
	       	        B[(j+0)*K+(i+3)]+=r03;
				}
			}
	        for(int j=M-M%4;j<M;j++)
	            for(int i=K-K%4;i<K;i++)
			        for(int k=0;k<N;k++)
						B[j*K+i]+=A[k*M+j]*Res[i*N+k];
	        for(int i=0;i<4;i++) delete[] packb[i];
	    }
	#endif
    void Matrix_Mul_old(int N, int M, int K, float* A, float* B, float* Res)
    {
        memset(Res, 0, sizeof(float) * K * N);
        for (int j = 0; j < K; j++)
        	for (int i = 0; i < N; i++)
           		for (int k = 0; k < M; k++)
           	        Res[j * N + i] += A[i * M + k] * B[k * K + j];
    }
    void Matrix_Mul_Back_B_old(int N, int M, int K, float* A, float* B, float* Res) // simply add
    {
        for (int k = 0; k < M; k++)
            for (int j = 0; j < K; j++)
		        for (int i = 0; i < N; i++)
					B[k * K + j] += A[i * M + k] * Res[j * N + i];
	}
    void Matrix_Mul_Back_A_old(int N, int M, int K, float* A, float* B, float* Res)
    {
        memset(A, 0, sizeof(float) * N * M);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < M; k++)
                for (int j = 0; j < K; j++)
                    A[i * M + k] += Res[j * N + i] * B[k * K + j];
    }
    void Matrix_Mul_Back_A(int N, int M, int K, float* A, float* B, float* Res){Matrix_Mul(M,K,N,B,Res,A);}
#endif
