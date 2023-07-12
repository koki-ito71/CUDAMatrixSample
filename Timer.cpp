#include"Timer.h"
#include"stdio.h"

Timer::Timer() {
	sum = 0;
	isActive = false;
}

void Timer::Start() {
	start = std::chrono::system_clock::now();
	isActive = true;
}

void Timer::Stop() {
	if (!isActive)return;
	auto end = std::chrono::system_clock::now();
	sum += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	isActive = false;
}

void Timer::Reset() {
	sum = 0;
	isActive = false;
}

double Timer::Elapsed() {
	if (!isActive)return sum;
	auto temp = std::chrono::system_clock::now();
	double delta = std::chrono::duration_cast<std::chrono::milliseconds>(temp - start).count();
	return sum + delta;
}

bool Timer::IsActive() {
	return isActive;
}