#ifndef ILL_DB_HPP_
#define ILL_DB_HPP_

const int kPhotonNum = 100;

struct PhotonRecord {
	int reflection_;
	int refraction_;
	double squ_dis_, r_, g_, b_, depth_;
};

struct HitPointRecord {
	double BRDF_, depth_;
	int reflection_;
	int refraction_;
	PhotonRecord photons_[kPhotonNum];
};

#endif // ILL_DB_HPP_
