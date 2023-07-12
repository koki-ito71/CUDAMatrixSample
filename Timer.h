#ifndef _Timer_
#define _Timer_

#include <chrono>


class Timer {
	std::chrono::system_clock::time_point start;

	double sum;

	bool isActive;

public: Timer();

public: void Start();

public: void Stop();

public: void Reset();

public: double Elapsed();

public: bool IsActive();

};

#endif
