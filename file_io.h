#pragma once

#include <fstream>
#include <vector>
#include <io.h>
#include <algorithm>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

template<typename T>
inline void readf(std::ifstream& inf, T &x){inf.read((char*)&x,sizeof(T));}
template<typename T>
inline void readf(std::ifstream& inf, T *x, int siz){inf.read((char*)x,sizeof(T)*siz);}

template<typename T>
inline void writf(std::ofstream& ouf, T x){ouf.write((char*)&x,sizeof(T));}
template<typename T>
inline void writf(std::ofstream& ouf, T *x, int siz){ouf.write((char*)x,sizeof(T)*siz);}

inline void readimg(std::string path,int &d,int &h,int &w,float *img,float l=-1,float r=1)
{
	int iw,ih,n;
	auto data=stbi_load(path.c_str(),&iw,&ih,&n,0);
	d=n,h=ih,w=iw;
	for(int i=0,p=0;i<w;i++)
		for(int j=0;j<h;j++)
			for(int k=0;k<d;k++) img[k*h*w+j*w+i]=data[p++]/(float)255*(r-l)+l;
	stbi_image_free(data);
}

inline void readimg(std::string path,float *img,float l=-1,float r=1)
{
	int d,h,w;
	readimg(path,d,h,w,img,l,r);
}

inline void savejpg(std::string path,int d,int h,int w,float *img,float l=-1,float r=1,
	int quality=100/*[1,100]*/)
{
	auto tmp=new unsigned char[w*h*d];
	for(int i=0,p=0;i<w;i++)
		for(int j=0;j<h;j++)
			for(int k=0;k<d;k++)
			{
				int x=(img[k*h*w+j*w+i]-l)/(r-l)*255;
				tmp[p++]=std::max(0,std::min(x,255));
			}
	stbi_write_jpg(path.c_str(),w,h,d,tmp,quality);
	delete[] tmp;
}

inline void savepng(std::string path,int d,int h,int w,float *img,float l=-1,float r=1)
{
	auto tmp=new unsigned char[w*h*d];
	for(int i=0,p=0;i<w;i++)
		for(int j=0;j<h;j++)
			for(int k=0;k<d;k++)
			{
				int x=(img[k*h*w+j*w+i]-l)/(r-l)*255;
				tmp[p++]=std::max(0,std::min(x,255));
			}
	stbi_write_png(path.c_str(),w,h,d,tmp,0);
	delete[] tmp;
}

inline void savebmp(std::string path,int d,int h,int w,float *img,float l=-1,float r=1)
{
	auto tmp=new unsigned char[w*h*d];
	for(int i=0,p=0;i<w;i++)
		for(int j=0;j<h;j++)
			for(int k=0;k<d;k++)
			{
				int x=(img[k*h*w+j*w+i]-l)/(r-l)*255;
				tmp[p++]=std::max(0,std::min(x,255));
			}
	stbi_write_bmp(path.c_str(),w,h,d,tmp);
	delete[] tmp;
}

void getfiles(std::string path,std::vector<std::string> &files)
{
    intptr_t hFile=0;
    struct _finddata_t fileinfo;
    std::string p;
    if((hFile=_findfirst(p.assign(path).append("\\*").c_str(),&fileinfo))!=-1)
    {
        do
        {
            if((fileinfo.attrib&_A_SUBDIR))
            {
                if(strcmp(fileinfo.name,".")!=0&&strcmp(fileinfo.name,"..")!=0)
                    getfiles(p.assign(path).append("\\").append(fileinfo.name),files);
            }
            else files.push_back(p.assign(path).append("\\").append(fileinfo.name));
        }while(_findnext(hFile,&fileinfo)==0);
        _findclose(hFile);
    }
}
