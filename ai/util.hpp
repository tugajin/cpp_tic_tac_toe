#ifndef __UTIL_HPP__
#define __UTIL_HPP__

// includes

#include <cstdint>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <vector>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>
#include <thread>
// constants

#undef FALSE
#define FALSE 0

#undef TRUE
#define TRUE 1

#ifdef DEBUG
#  undef DEBUG
#  define DEBUG TRUE
#else
#  define DEBUG FALSE
#endif


// macros

#if DEBUG
#  undef NDEBUG
#else
#ifndef NDEBUG
#  define NDEBUG
#endif
#endif

#include <cassert> // needs NDEBUG
#include <ctime>

// types

typedef std::int8_t  int8;
typedef std::int16_t int16;
typedef std::int32_t int32;
typedef std::int64_t int64;

typedef std::uint8_t  uint8;
typedef std::uint16_t uint16;
typedef std::uint32_t uint32;
typedef std::uint64_t uint64;

#if DEBUG

#define ASSERT(a) { if (!(a)) { Tee<<"file:"<<__FILE__<<" line:"<<__LINE__<<std::endl; std::exit(EXIT_FAILURE); }  }
#define ASSERT2(a,f) { if (!(a)) { Tee<<"file:"<<__FILE__<<" line:"<<__LINE__<<std::endl; f; std::exit(EXIT_FAILURE); }  }

#else
#define ASSERT(a) {}
#define ASSERT2(a,f) {}
#endif
std::string timestamp();

class TeeStream {
public:
	TeeStream() {
		ofs_.open("./log.log", std::ios::app);
		ofs_ << "LOG_START " + timestamp() << std::endl;
	}
	template<typename T>
	TeeStream& operator <<(const T& t) {
		if (level_ == 1) {
			lock_.lock();

			std::cout << t;
			ofs_ << t;

			ofs_.flush();
			std::cout.flush();

			lock_.unlock();
		}
		return *this;
	}
	TeeStream& operator <<(std::ostream& (*f)(std::ostream&)) {
		std::cout << f;
		ofs_ << f;
		return *this;
	}

private:
	std::ofstream ofs_;
	mutable std::mutex lock_;
	static constexpr int level_ = 1;
};

extern TeeStream Tee;

std::string timestamp() {
    std::time_t t = std::time(nullptr);
	char mbstr[256];
	if (std::strftime(mbstr, 256, "%Y%m%d-%H%M%S", std::localtime(&t))) {
	    std::string str = mbstr;
		return str;
	}
	return "";
}

uint64 rand_int_64() {
	std::random_device seed_gen;
  	std::mt19937_64 engine(seed_gen());
	return engine();
}

double rand_double() {
	std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(0, 1);
	return distr(eng);
}

double rand_gaussian(const double mean, const double variance) {
	std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> dist(mean, variance);
	return dist(engine);
}

int my_rand(int i) {
	return int(rand_int_64() % i);
}

std::string trim(const std::string s) {
	std::string str = s;
	const auto pos = str.find(' ');
	if (pos != std::string::npos) {
		str = str.substr(pos + 1);
	}
	return str;
}

std::string padding_str(std::string const &str, int n) {
    std::ostringstream oss;
    oss << std::setw(n) << str;
    return oss.str();
}

bool is_exists_file(const std::string path) {
	std::ifstream ifs(path);
	return ifs.is_open();
}

int my_choice(std::vector<int>score) {
	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	const auto sum_score =  std::accumulate(score.begin(), score.end(), 0);
	std::uniform_int_distribution<> dist(0, sum_score);
	int result = dist(engine);
	int sum = 0;
	for(auto i = 0u; i < score.size(); i++) {
		sum += score[i];
		if(sum > result) {
			return i;
		}
	}
	return score.size()-1;
} 

template<class T> std::string to_string(T x) {
	std::stringstream ss;
	ss << x;
	return ss.str();
}

void my_sleep(const int  millisec) {
	std::this_thread::sleep_for(std::chrono::milliseconds(millisec));
}

class Timer {

private:

	typedef std::chrono::time_point<std::chrono::system_clock> time_t;
	typedef std::chrono::duration<double> second_t;

	double elapsed_;
	bool running_;
	time_t start_;

public:

	Timer() {
		reset();
	}

	void reset() {
		elapsed_ = 0;
		running_ = false;
	}

	void start() {
		start_ = now();
		running_ = true;
	}

	void stop() {
		elapsed_ += time();
		running_ = false;
	}

	double elapsed() const {
		double time = elapsed_;
		if (running_) time += this->time();
		return time;
	}

private:

	static time_t now() {
		return std::chrono::system_clock::now();
	}

	double time() const {
		assert(running_);
		return std::chrono::duration_cast<second_t>(now() - start_).count();
	}
};

#endif