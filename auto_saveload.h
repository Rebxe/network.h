/*
   In your own class:
       -  Add AUTO_SL_BEG before all the declartions of the optimizer and layers
             and add AUTO_SL_END after all these declartions
       -  There must be only one optimizer, and it's declartion must be in front of
			   all the declartions of layers
   Example:
       class network
       {
       	AUTO_SL_BEG
       		ADAM opt;
       		FC fc1,fc2,fc3;
       		CONV c1,c2;
       	AUTO_SL_END
       }test;
       // now use test.save(path) to save
       // and use test.load(path,Batch_Size) to load
       // also use test.delthis() to free memories
*/

/*
   Use 
	 - AUTO_SL_LAYER_CONSTRUCTER_WEIGHT_DELTHISFUNC(classname)
	 - AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(classname)
	 - AUTO_SL_LAYER_CONSTRUCTER_DELTHISFUNC(classname)
	 - AUTO_SL_LAYER_CONSTRUCTER(classname)
   to declare construction function to enable auto save/load/delthis for your own layer
*/

/*
   Use
      AUTO_SL_OPTIMIZER_CONSTRUCTER(classname)
   to declare construction function to enable auto save/load/delthis for your own optimizer
*/

#pragma once

#include <vector>
#include <functional>
#include <fstream>
#include <vector>

namespace auto_sl
{
	typedef std::function<void(std::ofstream&)> SF;
	typedef std::function<void(std::ifstream&,float*&,float*&)> LF;
	typedef std::function<void()> DF;
	typedef std::function<float*()> GET_ADD;
	typedef std::vector<SF> VSF;
	typedef std::vector<LF> VLF;
	typedef std::vector<DF> VDF;
	
	std::vector<std::pair<
		std::pair<
			std::pair<VSF*,VLF*>,
			VDF*
		>,
		std::pair<GET_ADD*,GET_ADD*>
	> > stk;
	
	struct STRUCT_LAYERS_SL_BEG
	{
		VSF vsf;
		VLF vlf;
		VDF vdf;
		GET_ADD _wei,_tmp;
		inline STRUCT_LAYERS_SL_BEG()
		{
			stk.emplace_back(
				std::make_pair(
					std::make_pair(&vsf,&vlf),
					&vdf
				),
				std::make_pair(&_wei,&_tmp)
			);
		}
	};
	struct STRUCT_LAYERS_SL_END_DELTHIS
	{
		VDF *vdf;
		inline STRUCT_LAYERS_SL_END_DELTHIS(){vdf=stk.back().first.second;}
		inline void operator()(){for(auto f:*vdf) f();}
	};
	struct STRUCT_LAYERS_SL_END_SAVE
	{
		VSF *vsf;
		inline STRUCT_LAYERS_SL_END_SAVE(){vsf=stk.back().first.first.first;}
		inline void operator()(std::string path)
		{
			std::ofstream ouf(path,std::ios::out|std::ios::binary); 
			for(auto f:*vsf) f(ouf);
			ouf.close();
		}
	};
	struct STRUCT_LAYERS_SL_END_LOAD
	{
		VLF *vlf;
		GET_ADD *_wei,*_tmp;
		inline STRUCT_LAYERS_SL_END_LOAD()
		{
			vlf=stk.back().first.first.second;
			_wei=stk.back().second.first;
			_tmp=stk.back().second.second;
			stk.pop_back();
		}
		inline void operator()(std::string path)
		{
			std::ifstream inf(path,std::ios::in|std::ios::binary);
			float *_1=NULL,*_2=NULL;
			(*vlf)[0](inf,_1,_2);
			float *wei=(*_wei)(),*tmp=(*_tmp)();
			for(int i=1;i<(*vlf).size();i++) (*vlf)[i](inf,wei,tmp);
			inf.close();
		}
	};
	#define AUTO_SL_BEG auto_sl::STRUCT_LAYERS_SL_BEG __STRUCT_LAYERS_SL__;
	#define AUTO_SL_END auto_sl::STRUCT_LAYERS_SL_END_DELTHIS delthis;\
						auto_sl::STRUCT_LAYERS_SL_END_SAVE save;\
						auto_sl::STRUCT_LAYERS_SL_END_LOAD load;
}

#define AUTO_SL_LAYER_CONSTRUCTER_WEIGHT_DELTHISFUNC(name) inline name()\
{\
	auto savef = std::bind(\
		&std::remove_reference<decltype(*this)>::type::save,\
		this,\
		std::placeholders::_1);\
	auto tmp = std::bind(\
		&std::remove_reference<decltype(*this)>::type::load,\
		this,\
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);\
	auto loadf = std::bind(\
		tmp,\
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3\
		);\
	auto_sl::stk.back().first.first.first->push_back(savef);\
	auto_sl::stk.back().first.first.second->push_back(loadf);\
	auto delf = std::bind(\
		&std::remove_reference<decltype(*this)>::type::delthis,\
		this\
		);\
	auto_sl::stk.back().first.second->push_back(delf);\
}

#define AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(name) inline name()\
{\
	auto savef = std::bind(\
		&std::remove_reference<decltype(*this)>::type::save,\
		this,\
		std::placeholders::_1);\
	auto tmp = std::bind(\
		&std::remove_reference<decltype(*this)>::type::load,\
		this,\
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);\
	auto loadf = std::bind(\
		tmp,\
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3\
		);\
	auto_sl::stk.back().first.first.first->push_back(savef);\
	auto_sl::stk.back().first.first.second->push_back(loadf);\
}

#define AUTO_SL_LAYER_CONSTRUCTER_DELTHISFUNC(name) inline name()\
{\
	auto savef = std::bind(\
		&std::remove_reference<decltype(*this)>::type::save,\
		this,\
		std::placeholders::_1);\
	auto tmp = std::bind(\
		&std::remove_reference<decltype(*this)>::type::load,\
		this,\
		std::placeholders::_1);\
	auto loadf = std::bind(\
		tmp,\
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3\
		);\
	auto_sl::stk.back().first.first.first->push_back(savef);\
	auto_sl::stk.back().first.first.second->push_back(loadf);\
	auto delf = std::bind(\
		&std::remove_reference<decltype(*this)>::type::delthis,\
		this\
		);\
	auto_sl::stk.back().first.second->push_back(delf);\
}

#define AUTO_SL_LAYER_CONSTRUCTER(name) inline name()\
{\
	auto savef = std::bind(\
		&std::remove_reference<decltype(*this)>::type::save,\
		this,\
		std::placeholders::_1);\
	auto tmp = std::bind(\
		&std::remove_reference<decltype(*this)>::type::load,\
		this,\
		std::placeholders::_1);\
	auto loadf = std::bind(\
		tmp,\
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3\
		);\
	auto_sl::stk.back().first.first.first->push_back(savef);\
	auto_sl::stk.back().first.first.second->push_back(loadf);\
}

#define AUTO_SL_OPTIMIZER_CONSTRUCTER(name) inline name()\
{\
	auto savef = std::bind(\
		&std::remove_reference<decltype(*this)>::type::save,\
		this,\
		std::placeholders::_1);\
	auto tmp = std::bind(\
		&std::remove_reference<decltype(*this)>::type::load,\
		this,\
		std::placeholders::_1);\
	auto loadf = std::bind(\
		tmp,\
		std::placeholders::_1, std::placeholders::_2, std::placeholders::_3\
		);\
	auto_sl::stk.back().first.first.first->push_back(savef);\
	auto_sl::stk.back().first.first.second->push_back(loadf);\
	auto delf = std::bind(\
		&std::remove_reference<decltype(*this)>::type::delthis,\
		this\
		);\
	auto_sl::stk.back().first.second->push_back(delf);\
	(*auto_sl::stk.back().second.first) = std::bind(\
		&std::remove_reference<decltype(*this)>::type::_wei,\
		this);\
	(*auto_sl::stk.back().second.second) = std::bind(\
		&std::remove_reference<decltype(*this)>::type::_tmp,\
		this);\
}
