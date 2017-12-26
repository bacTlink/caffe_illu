#include <bits/stdc++.h>
#include "../include/illu/ill_db.hpp"
using namespace std;

int N, P, H, W;

struct Pos {
	double x_, y_, z_;
};

struct TmpPhotonRecord {
	int reflection_;
	int refraction_;
	double r_, g_, b_, depth_;
	Pos pos;
} photon[1000000];

inline double sqr(const double& x) { return x * x; }
double squ_dis(const Pos& p1, const Pos& p2) {
	return sqr(p1.x_ - p2.x_) + sqr(p1.y_ - p2.y_) + sqr(p1.z_ - p2.z_);
}

int id[100000];

struct Comparator {
	Comparator(Pos p): p_(p) {}
	bool operator () (const int& id1, const int& id2) {
		return squ_dis(photon[id1].pos, p_) < squ_dis(photon[id2].pos, p_);
	}
	Pos p_;
};

PhotonRecord getPhotonRecord(const TmpPhotonRecord& tmp, Pos pos) {
	PhotonRecord res;
	res.reflection_ = tmp.reflection_;
	res.refraction_ = tmp.refraction_;
	res.squ_dis_ = squ_dis(pos, tmp.pos);
	res.r_ = tmp.r_;
	res.g_ = tmp.g_;
	res.b_ = tmp.b_;
	res.depth_ = tmp.depth_;
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		printf("Usage: data_pretreat filename\n");
		return 0;
	}
	FILE* fin = freopen(argv[1], "r", stdin);
	scanf("%d", &N);
	FILE* data_list = fopen("data_list.txt", "w");
	while (N--) {
		char filename[100];
		scanf("%s", filename);
		assert(string(filename).substr(strlen(filename) - 4) == ".bmp");
		string datafilename = filename;
		datafilename.replace(strlen(filename) - 4, 4, ".dat");

		fprintf(data_list, "%s\n", datafilename.c_str());

		FILE* datafile = fopen(datafilename.c_str(), "wb");
		fwrite(filename, sizeof(char), 100, datafile);

		scanf("%d", &P);
		for (int i = 0; i < P; ++i)
			scanf("%lf%lf%lf%lf%lf%lf%d%d%lf", &photon[i].pos.x_, &photon[i].pos.y_, &photon[i].pos.z_, &photon[i].r_, &photon[i].g_, &photon[i].b_, &photon[i].reflection_, &photon[i].refraction_, &photon[i].depth_);

		scanf("%d%d", &H, &W);
		fwrite(&H, sizeof(int), 1, datafile);
		fwrite(&W, sizeof(int), 1, datafile);

		HitPointRecord res;
		Pos pos;
		for (int i = 0; i < H; ++i)
			for (int j = 0; j < W; ++j) {
				scanf("%lf%lf%lf%lf%d%d%lf", &res.BRDF_, &pos.x_, &pos.y_, &pos.z_, &res.reflection_, &res.refraction_, &res.depth_);
				int M;
				scanf("%d", &M);
				memset(res.photons_, 0, sizeof(res.photons_));
				for (int p = 0; p < M; ++p)
					scanf("%d", id + p);
				sort(id, id + M, Comparator(pos));
				for (int p = 0; p < kPhotonNum; ++p)
					res.photons_[p] = getPhotonRecord(photon[id[p]] ,pos);
				fwrite(&res, sizeof(HitPointRecord), 1, datafile);
			}

		fclose(datafile);
	}
	fclose(data_list);
	fclose(fin);
	return 0;
}
