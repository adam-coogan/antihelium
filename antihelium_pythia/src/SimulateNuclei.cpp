//-------------------------------------------------------------------------
// main01.cc
// Author:Eric Carlson.
// Run pythia for several different DM annihilation processes and check
// check formation rates of anti-nuclei up to A=4
//-------------------------------------------------------------------------
#include <iostream>
#include <string>
#include <sstream>
#include <ctime>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <random>
//#include <boost/thread.hpp>
//#include <boost/thread/mutex.hpp>
#include <omp.h>

#include "Pythia8/Pythia.h"
#include <fstream>
//#include "Miniball.hpp"

using namespace Pythia8;
using namespace std;


//------------------------------------------------------------------------
// Function Prototypes
//------------------------------------------------------------------------
double computeDistance(Vec4, Vec4);
bool checkCoal(int numParticles, Vec4 p[]);
void writeEvent(double CMS, int numParticles, Particle part[], double pCoal);
int main(int argc, char *argv[]);
void analyzeEvent(double CMS, Event event, int seed);
void pythiaThread(int numEvents, double CMS, int seed, int process);

Vec4 getCentroid(Vec4, Vec4);

//------------------------------------------------------------------------
// Global Declarations
//------------------------------------------------------------------------
int antideuteron = 0; // Number of antideuterons
float antideuteron_cross_sec = 0; // Number of antideuterons
int antihelium3  = 0;
int antihelium4  = 0;
bool cross_sec_method = true;

//------------------------------------------------------------------------
// Settings
//------------------------------------------------------------------------
float pCoalHigh = .3; // Don't record distances more than x.xxx GeV
float sigma_0   =  2.13e-6; // inverse cross-section sigma_0 in inverse microbarn
//long totalEvents = 1e6;  // use command line arg.
bool ISOSPIN_TEST = false; // Output protons and neutrons? **NOT IMPLEMENTED**


//------------------------------------------------------------------------
// Some PDG Particle Codes
const int PDG_pbar = -2212; // Antiproton
const int PDG_nbar = -2112; // Antineutron
const int PDG_e    = 11; // electron
const int PDG_ebar = -11; // electron
const int PDG_b    = 5;
const int PDG_bbar = -5;



// Filewriter
ofstream  pbar_eventFile;
ofstream  dbar_eventFile;
ofstream  heavy_eventFile;
ofstream  ww_eventFile;

// mutex lock for file writeer
//boost::mutex mutex_dbar;
//boost::mutex mutex_heavy;
//--------------------------------------------------------------
// OpenMP mutex code
//--------------------------------------------------------------
struct MutexType
 {
   MutexType() { omp_init_lock(&lock); }
   ~MutexType() { omp_destroy_lock(&lock); }
   void Lock() { omp_set_lock(&lock); }
   void Unlock() { omp_unset_lock(&lock); }

   MutexType(const MutexType& ) { omp_init_lock(&lock); }
   MutexType& operator= (const MutexType& ) { return *this; }
 public:
   omp_lock_t lock;
 };

struct ScopedLock
 {
   explicit ScopedLock(MutexType& m) : mut(m), locked(true) { mut.Lock(); }
   ~ScopedLock() { Unlock(); }
   void Unlock() { if(!locked) return; locked=false; mut.Unlock(); }
   void LockAgain() { if(locked) return; mut.Lock(); locked=true; }
 private:
   MutexType& mut;
   bool locked;
 private: // prevent copying the scoped lock.
   void operator=(const ScopedLock&);
   ScopedLock(const ScopedLock&);
 };

MutexType lock_dbar;
MutexType lock_heavy;
MutexType lock_ww;
MutexType lock_pbar;




// A derived class for (e+ e- ->) GenericResonance -> various final states.

class Sigma1GenRes : public Sigma1Process {

public:

  // Constructor.
  Sigma1GenRes() {}

  // Evaluate sigmaHat(sHat): dummy unit cross section. 
  virtual double sigmaHat() {return 1.;}

  // Select flavour. No colour or anticolour.
  virtual void setIdColAcol() {setId( -11, 11, 999999);
    setColAcol( 0, 0, 0, 0, 0, 0);}

  // Info on the subprocess.
  virtual string name()    const {return "GenericResonance";}
  virtual int    code()    const {return 9001;}
  virtual string inFlux()  const {return "ffbarSame";}

};

/////////////////////////////////////////////////////////////////
//  Utility                                                    //
/////////////////////////////////////////////////////////////////

// 3-vector class
struct vec3
{
    vec3(){}
    vec3(float v0, float v1, float v2){vals[0]=v0;vals[1]=v1;vals[2]=v2;}
    vec3(float v0){vals[0]=v0;vals[1]=v0;vals[2]=v0;}    
    float vals[3];
    float length() const{return sqrt(vals[0]*vals[0]+vals[1]*vals[1]+vals[2]*vals[2]);}
    float& operator[](unsigned int i){ return vals[i];}     
    const float operator[](unsigned int i)const{ return vals[i];}       
};

vec3 operator* (float x, const vec3 &y)
{
    vec3 retVec = y;
    for(int i=0;i<3;i++)
    {
        retVec[i] *= x;
    }
    return retVec;
}
vec3 operator* (const vec3 &y, float x)
{
    vec3 retVec = y;
    for(int i=0;i<3;i++)
    {
        retVec[i] *= x;
    }
    return retVec;
}
vec3 operator- (const vec3 &x)
{
    vec3 retVec;
    for(int i=0;i<3;i++)
    {
        retVec[i] = -x[i];
    }
    return retVec;
}
vec3 operator- (const vec3 &x, const vec3 &y)
{
    vec3 retVec = x;
    for(int i=0;i<3;i++)
    {
        retVec[i] -= y[i];
    }
    return retVec;
}
vec3 operator+ (const vec3 &x, const vec3 &y)
{
    vec3 retVec = x;
    for(int i=0;i<3;i++)
    {
        retVec[i] += y[i];
    }
    return retVec;
}
vec3 operator/ (const vec3 &x, const float &y)
{
    vec3 retVec = x;
    for(int i=0;i<3;i++)
    {
        retVec[i] /= y;
    }
    return retVec;
}

// 4-vector class
struct vec4
{
    vec4(){}
    vec4(float v0, float v1, float v2, float v3){vals[0]=v0;vals[1]=v1;vals[2]=v2;vals[3]=v3;} 
    vec4(float v0){vals[0]=v0;vals[1]=v0;vals[2]=v0;vals[3]=v0;}   
    vec4(float v0, vec3 v){vals[0]=v0;vals[1]=v[0];vals[2]=v[1];vals[3]=v[2];}   
    float vals[4];
    float& operator[](unsigned int i){ return vals[i];}     
    float operator[](unsigned int i)const{ return vals[i];}   
    vec3 xyz() const{return vec3(vals[1],vals[2],vals[3]);}     
};

vec4 operator- (const vec4 &x, const vec4 &y)
{
    vec4 retVec = x;
    for(int i=0;i<4;i++)
    {
        retVec[i] -= y[i];
    }
    return retVec;
}
vec4 operator+ (const vec4 &x, const vec4 &y)
{
    vec4 retVec = x;
    for(int i=0;i<4;i++)
    {
        retVec[i] += y[i];
    }
    return retVec;
}
double momentum(vec4 in)
{
    return sqrt(in[1]*in[1]+in[2]*in[2]+in[3]*in[3]);
}

// 4x4 matrix class
struct mat4
{
    double vals[4][4];
    mat4(){};
    mat4(   float v00, float v01, float v02, float v03,
            float v10, float v11, float v12, float v13,
            float v20, float v21, float v22, float v23,
            float v30, float v31, float v32, float v33)
    {
        vals[0][0] = v00;
        vals[0][1] = v01;
        vals[0][2] = v02;
        vals[0][3] = v03; 
        vals[1][0] = v10;
        vals[1][1] = v11;
        vals[1][2] = v12;
        vals[1][3] = v13;
        vals[2][0] = v20;
        vals[2][1] = v21;
        vals[2][2] = v22;
        vals[2][3] = v23;
        vals[3][0] = v30;
        vals[3][1] = v31;
        vals[3][2] = v32;
        vals[3][3] = v33;
    }
    mat4(double v)
    {
        vals[0][0] = v;
        vals[0][1] = v;
        vals[0][2] = v;
        vals[0][3] = v; 
        vals[1][0] = v;
        vals[1][1] = v;
        vals[1][2] = v;
        vals[1][3] = v;
        vals[2][0] = v;
        vals[2][1] = v;
        vals[2][2] = v;
        vals[2][3] = v;
        vals[3][0] = v;
        vals[3][1] = v;
        vals[3][2] = v;
        vals[3][3] = v;
    } 
    // Identity matrix   
    static mat4 identity()
    {
        return mat4(1,0,0,0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1);
    }    
};

vec4 operator* (const mat4 &m, const vec4 &v)
{
    vec4 out(0);
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            out[i] += m.vals[i][j]*v[j];
        }
    }
    return out;
}
mat4 operator* (const mat4 &m1, const mat4 &m2)
{
    mat4 out(0);
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            for(int k=0; k<4; k++)
            {
                out.vals[i][j] += m1.vals[i][k]*m2.vals[k][j];
            }
        }
    }
    return out;
}
double dot(const vec3 &a, const vec3 &b)
{
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
double dot(const vec4 &a, const vec4 &b)
{
    return a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
}
// Calculate Lorentz boost matrix corresponding to beta_xyz
void lorentzMatrix(const vec3 &beta_xyz, mat4 &mat)
{
    double b = beta_xyz.length();
    double bm2 = b==0 ? 0 : 1.0/(b*b);
    double bx = beta_xyz[0];
    double by = beta_xyz[1];
    double bz = beta_xyz[2];                 
    double g = 1.0/sqrt(1-b*b);
    mat =  mat4(    g,      -g*bx,              -g*by,              -g*bz,
                    -g*bx,  1+(g-1)*bx*bx*bm2,  (g-1)*bx*by*bm2,    (g-1)*bx*bz*bm2,
                    -g*by,  (g-1)*by*bx*bm2,    1+(g-1)*by*by*bm2,  (g-1)*by*bz*bm2,
                    -g*bz,  (g-1)*bz*bx*bm2,    (g-1)*bz*by*bm2,    1+(g-1)*bz*bz*bm2);  
}
// Generate a random number between -1 and 1
double rand_m1_1()
{
    return -1.0 + 2.0 * static_cast<double>(rand()) / RAND_MAX;
}
// Generate a random number between 0 and 1
double rand_0_1()
{
    return static_cast<double>(rand()) / RAND_MAX;
}

/////////////////////////////////////////////////////////////////
//  Functions for generating phase space                       //
/////////////////////////////////////////////////////////////////

// Calculate Mandelstam variable s. 
double calc_s(const vec4 &p1, const vec4 &p2)
{
    vec4 sum = p1+p2;
    return dot(sum,sum);
}
// Generate a 3-vector to a random point on the unit sphere
vec3 randOnSphere()
{
    double r1,r2;
    do
    {
        r1 = rand_m1_1();
        r2 = rand_m1_1();
    }
    while(r1*r1+r2*r2 >=1.0);
    vec3 v;
    v[0] = 2.0*r1*sqrt(1-r1*r1-r2*r2);
    v[1] = 2.0*r2*sqrt(1-r1*r1-r2*r2);
    v[2] = 1.0-2.0*(r1*r1+r2*r2);
    return v;
}
// Boost antideuteron to lab frame. Takes nucleon lab momenta and antideuteron CM momentum as input.
vec4 boostToLabFrame(vec4 p_N1_lab, vec4 p_N2_lab, vec4 p_d_CM)
{
    vec3 beta_CM = (p_N1_lab.xyz() + p_N2_lab.xyz())/(p_N1_lab[0]+p_N2_lab[0]);   
    mat4 boost_CM_lab;          
    lorentzMatrix(-beta_CM,boost_CM_lab);   
    return boost_CM_lab * p_d_CM;
}
// Phase space generation for 3-body decay.
// Input: Mandelstam variable s, and masses of the 3 particles. 
// Returns energy of the third particle in the CM frame, randomly drawn from allowed phase space (assuming no angular correlations).
double dalitz(double s, double m1, double m2, double m3)
{
    double& Msq = s;
    double M = sqrt(Msq);
    double m1sq = m1*m1;
    double m2sq = m2*m2;
    double m3sq = m3*m3;
    double m12sqMin = pow(m1+m2, 2);
    double m12sqMax = pow(M-m3,  2);
    double m12sq;
    double m23sq;
    while(true)
    {
        m12sq = m12sqMin+rand_0_1()*(m12sqMax-m12sqMin);
        double m12 = sqrt(m12sq);
        double _E2 = 0.5*(m12sq-m1sq+m2sq)/m12;
        double _E3 = 0.5*(Msq-m12sq-m3sq)/m12;
        double m23sqMin = pow(m2+m3, 2);
        double m23sqMax = pow(M-m1 , 2);
        double tmp1 = pow(_E2+_E3, 2);
        double _p2 = sqrt(_E2*_E2-m2sq);
        double _p3 = sqrt(_E3*_E3-m3sq);
        double m23sqU = tmp1 - pow(_p2-_p3 ,2);
        double m23sqL = tmp1 - pow(_p2+_p3 ,2);
        m23sq = m23sqMin+rand_0_1()*(m23sqMax-m23sqMin);
        if(m23sq <= m23sqU and m23sq >= m23sqL) break;
    }
    return 0.5*(Msq+m3sq-m12sq)/M;
}


// --------------------------------------------------------------------
// Taken from pg 3 of Raklev & Dal 2015 1504.07242v1.  Table II + Eqn 7
double a_coef [12] = {2.30346, -9.366346e1, 2.565390e3, -2.5594101e4, 1.43513109e5, -5.0357289e5,
			 1.14924802e6, -1.72368391e6, 1.67934876e6, -1.01988855e6, 3.4984035e5, -5.1662760e4};
double b_1 = -5.1885;
double b_2 = 2.9196;
double eq7(double k){
	// param k is relative momenta of p,n pair in GeV/c
	double sum = 0; 
	if (k < 1.28){
		for (int i=0; i<12; i++){
			sum += a_coef[i]*pow(k,i-1)	; 
		}
	}
	else{
		sum = exp(-b_1*k-b_2*k*k);
	}
	return sum; 
}

// Taken from pg 4 of Raklev & Dal 2015 1504.07242v1.  Table III + Eqn 10
double eq10(double k){
	// param q_pi where q is pion momentum in COM frame in MeV
	double m_pi_plus = 139.570182; 
	double q_pi = sqrt(k*k*1e6-m_pi_plus*m_pi_plus)/2.;
	if (q_pi < 0)
		return 0;

	double eta = q_pi/139.57018;
	return 170*pow(eta,1.34)/(pow(1.77-exp(0.38*eta),2.)+.096);
}

// Taken from Dal & Raklev 2015 1504.07242v1.  Eqn 13
double eq13_pi0pi0(double k){
	if (k < .139570182) 
		return 0.;
	return 2.855e6*pow(k,1.311e1)/(pow(2.961e3-exp(5.572*k),2)+1.461e6);
}

double eq13_pi0piplus(double k){
	if (k < .13957018*2) 
		return 0.;
	return 5.099e15*pow(k,1.656e1)/(pow(2.333e7-exp(1.133e1*k),2)+2.868e16);
}

// Taken from Dal & Raklev 2015 1504.07242v1.  Eqn 14
double eq14(double k){
	if (k < .13957018*2) 
		return 0.;
	return 6.465e6*pow(k,1.051e1)/(pow(1.979e3-exp(5.363*k),2)+6.045e5)
		 + 2.579e15*pow(k,1.657e1)/(pow(2.330e7-exp(1.119e1*k),2)+2.868e16);
}




// Particle masses (GeV)
const double mn = 0.93956536;   // Neutron
const double mp = 0.93827203;   // Proton
const double md = 1.875612793;  // Deuteron
const double mpic = 0.13957018; // Charged pion
const double mpi0 = 0.1349766;  // Neutral pion

// Fit function used for pion processes
double fitFunc(double x, double a, double b, double c, double d, double e)
{
    return a*pow(x,b)/(pow((c-exp(d*x)),2)+e);
}

// N N -> d pi helper function
double xs_pp_dpip_q(double q)
{
    double eta = q/mpic;
    double a[5] = {0.17, 1.34, 1.77, 0.38, 0.096};    
    return 1e3 * fitFunc(eta, a[0],a[1],a[2],a[3],a[4]);
}

/////////////////////////////////////////////////////////////////
//  Cross section parameterizations                            //
/////////////////////////////////////////////////////////////////

// p n -> d gamma
// Returns cross section in microbarn. k must be in units of GeV.
double xs_pn_dgamma(double k)
{
    double a[12] = {2.3034605532591175,  -93.663463313902028, 2565.3904680353621, 
                    -25594.100560137995, 143513.10872427333,  -503572.89020794741, 
                    1149248.0196165806,  -1723683.9119787284, 1679348.7891145353, 
                    -1019888.5470232342, 349840.35161061864,  -51662.760038375141};
    double b[2]  = {-5.1885266705385051, 2.9195632726211609};
    if(k<1.28)
    {
        double result = 0;        
        for(int i=0;i<12;i++)
        {
            result += a[i] * pow(k,i-1);
        }
        return result;        
    }
    else
    {
        return exp(-b[0]*k -b[1]*k*k);
    }
}

// p n -> d pi0
// Returns cross section in microbarn. k must be in units of GeV.
double xs_pn_dpi0(double k)
{
    double E_CoM = sqrt(mp*mp+0.25*k*k) + sqrt(mn*mn+0.25*k*k);
    double s = E_CoM*E_CoM;
    if(E_CoM < md+mpi0) 
        return 0;
    double q = sqrt(0.25*pow(s+mpi0*mpi0-md*md,2)/s - mpi0*mpi0);
    return 0.5*xs_pp_dpip_q(q);
}
// p n -> d pi+ pi-
// Returns cross section in microbarn. k must be in units of GeV.
double xs_pn_dpippim(double k)
{
    double E_CoM = sqrt(mp*mp+0.25*k*k) + sqrt(mn*mn+0.25*k*k);
    if(E_CoM < md+2*mpic) 
        return 0;    
    double a[10] = {6.46455516e+06, 1.05136338e+01, 1.97924778e+03, 5.36301369e+00, 6.04534114e+05, 2.54935423e+15, 1.65669163e+01, 2.32961298e+07, 1.11937373e+01, 2.86815089e+16};
    return fitFunc(k, a[0],a[1],a[2],a[3],a[4]) + fitFunc(k, a[5],a[6],a[7],a[8],a[9]);    
}
// p n -> d pi0 pi0
// Returns cross section in microbarn. k must be in units of GeV.
double xs_pn_dpi0pi0(double k)
{
    double E_CoM = sqrt(mp*mp+0.25*k*k) + sqrt(mn*mn+0.25*k*k);
    if(E_CoM < md+2*mpi0) 
        return 0;     
    double a[5] = {2.85519622e+06, 1.31114126e+01, 2.96145497e+03, 5.57220777e+00, 1.46051932e+06};
    return fitFunc(k, a[0],a[1],a[2],a[3],a[4]);
}
// p p -> d pi+
// Returns cross section in microbarn. k must be in units of GeV.
double xs_pp_dpip(double k)
{
    double E_CoM = 2*sqrt(mp*mp+0.25*k*k);
    double s = E_CoM*E_CoM;
    if(E_CoM < md+mpic) 
        return 0;
    double q = sqrt(0.25*pow(s+mpic*mpic-md*md,2)/s - mpic*mpic);
    return xs_pp_dpip_q(q);
}
// p p -> d pi+ pi0
// Returns cross section in microbarn. k must be in units of GeV.
double xs_pp_dpippi0(double k)
{
    double E_CoM = 2*sqrt(mp*mp+0.25*k*k);
    if(E_CoM < md+mpic+mpi0) 
        return 0;     
    double a[5] = {5.09870846e+15, 1.65581228e+01, 2.33337076e+07, 1.13304315e+01, 2.86815089e+16};
    return fitFunc(k, a[0],a[1],a[2],a[3],a[4]);    
}
// n n -> d pi-
// Returns cross section in microbarn. k must be in units of GeV.
double xs_nn_dpim(double k)
{
    double E_CoM = 2*sqrt(mn*mn+0.25*k*k);
    double s = E_CoM*E_CoM;
    if(E_CoM < md+mpic) 
        return 0;  
    double q = sqrt(0.25*pow(s+mpic*mpic-md*md,2)/s - mpic*mpic);
    return xs_pp_dpip_q(q);
}
// n n -> d pi- pi0
// Returns cross section in microbarn. k must be in units of GeV.
double xs_nn_dpimpi0(double k)
{
    double E_CoM = 2*sqrt(mn*mn+0.25*k*k);
    if(E_CoM < md+mpic+mpi0) 
        return 0;      
    double a[5] = {5.09870846e+15, 1.65581228e+01, 2.33337076e+07, 1.13304315e+01, 2.86815089e+16};
    return fitFunc(k, a[0],a[1],a[2],a[3],a[4]);   
}






/////////////////////////////////////////////////////////////////////////
// main() starts each thread.  Uses boost libraries for multi-threading
/////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {

	if (argc != 7){
		cout << "\nSYNTAX:\nmain01.exe <Event File> <Process> <numCPU> <numNodes> <total events> <m_chi (GeV)>" << endl;
		cout << "Processes:\n0: e+ e- -> z -> b bbar\n1: e+ e- -> z -> t tbar\n2: e+ e- -> z -> w+ w-\n3: e+ e- -> g g (NOT IMPLEMENTED YET)\n4: e+ e- -> h z "<< endl;
		return 0;
	}

	//			0: e+ e- -> z -> b bbar
	//			1: e+ e- -> z -> t tbar
	// 			2: e+ e- -> z -> w+ w-
	//			3: e+ e- -> g g // NOT IMPLEMENTED
	//			4: e+ e- -> h z
	int process = atoi(argv[2]);
	int numCPU = atoi(argv[3]);  // Uncomment to specify num CPUs
	int numNodes = atoi(argv[4]);
	int totalEvents = atoi(argv[5]); // 
	float CMS = 2*atoi(argv[6]); //

	int numEvents = (int) (totalEvents/numNodes); // Total number of events.  Will be distributed over threads and nodes
	cout <<"TESTS"<<endl;
	int seed = (int) (getpid());

	// Find number of CPUs
	//long numCPU = sysconf( _SC_NPROCESSORS_ONLN );

	cout << "Using " << numCPU << " CPUs..." << endl;


	// Initialize file writer
	cout << string(argv[1]) + "_dbar" << endl;

	std::stringstream ss;
	//ss << seed;


	dbar_eventFile.open((string(argv[1]) + 	ss.str() +  "_dbar").c_str());
	heavy_eventFile.open((string(argv[1]) +	ss.str() + "_heavy").c_str());
    //ww_eventFile.open((string(argv[1]) +	ss.str() + "_ww").c_str());
	pbar_eventFile.open((string(argv[1]) +	ss.str() + "_pbar").c_str());

	
	// double CMS[3] = {100,500,1000};   // for b bbar // mid was 2000 before, redundant, needed 200
	// // Different masses for ww
	// if (process==2){
	// 		CMS[0] = 200;// for W+ W-
 //            CMS[1] = 2000;// for W+ W-
	// 		//CMS[2] = 2000; // Just changed from 1000
	// 		CMS[2] =4000;
	// }

	// int startIDX = 0;
	// int endIDX = 3; // in case we dont want to run all masses

	//for (int massidx=startIDX; massidx < endIDX; massidx++){
		// Reinit the RNG
		srand(time(NULL)*seed);
        long seeds[1000] = {};
        for (int i = 0 ; i < numCPU ; i++){
            seeds[i] = (long) (abs((i+1)*900000000./numCPU*rand())); // this 9e8 is max seed to pass to RNG

        }

		// Setup OpenMP
		#pragma omp parallel num_threads(numCPU)
		// Run parallel for loop
		#pragma omp for

		for (int i = 0 ; i < numCPU ; i++){
            cout << "SEED THREAD " << i << " = " << seeds[i] << endl;
			pythiaThread( (int) (numEvents/numCPU), CMS, seeds[i],process);
		}// end for i
	//}// end for massidx
	// Close filewriter
	dbar_eventFile.close();
	heavy_eventFile.close();
	//ww_eventFile.close();
	pbar_eventFile.close();
	
	cout << "Job Finsished!" << endl;
	return 1; // 1 for success
} // End Main



//////////////////////////////////////////////////////////////////////////
// pythiaThread starts a new instance of pythia for each thread.  This is
// where pythia should be configured.
// input:
//	-numEvents: number of events to simulate
//  -CMS: Center of mass energy
//  -seed: random number seed
//	-process:
//			0: e+ e- -> z -> b bbar
//			1: e+ e- -> z -> t tbar
// 			2: e+ e- -> z -> w+ w-
//			3: e+ e- -> g g // NOT IMPLEMENTED
//			4: e+ e- -> h z
//////////////////////////////////////////////////////////////////////////
void pythiaThread(int numEvents, double CMS, int seed, int process)
{
	// Generator. Process selection. LHC initialization. Histogram.
	Pythia pythia;

    SigmaProcess* sigma1GenRes = new Sigma1GenRes();

    // Hand pointer to Pythia.
    pythia.setSigmaPtr( sigma1GenRes);

	// Output Message every 10,000 events
	pythia.readString("Next:numberCount = 50000.");

	// Electron-Positron Collisions
	std::stringstream s;
	std::stringstream partInit;
	s << "Beams:eCM = " << CMS;
	
	pythia.readString("PartonLevel:ISR=off");
    pythia.readString("PDF:lepton = off"); // NO ISR
	pythia.readString(s.str()); // CMS Energy in GeV
	pythia.readString("Beams:idA = 11");
	pythia.readString("Beams:idB = -11");

    // Setup p-pbar collider
    //pythia.readString("Beams:idA = 2212");
    //pythia.readString("Beams:idB = -2212");

    std::stringstream seed_str;
    seed_str << "Random:seed = " << seed;

	// Init Random number generator
    pythia.readString("Random:setSeed = on");
    pythia.readString(seed_str.str());
	pythia.rndm.init(seed); // Generate random seed for pseudo-random generator);

	//---------------------------------------------------------------
	// PROCESS SETTINGS
    //! id:all = name antiName spinType chargeType colType m0 mWidth mMin mMax tau0
    partInit << "999999:all = GeneralResonance void 1 0 0 " << CMS << " 1. 0. 0. 0.";
    pythia.readString(partInit.str());
//! id:all = name antiName spinType chargeType colType m0 mWidth mMin mMax tau0
    
	//---------------------------------------------------------------
	// e+ e- -> z -> b bbar
	if (process == 0){
	    pythia.readString("999999:addChannel = 1 1. 101 5 -5");
	
		// Just turn on the weak single boson generation.
		//pythia.readString("WeakSingleBoson:ffbar2ffbar(s:gm) = on");
		// Setup channels
		//pythia.readString("23:oneChannel = 1 1. 0 5");// Set Z (23) to just b b-bar decay products. Bright-Wigner tail of Z dominates at mass > ~91 GeV.
		//pythia.readString("22:onMode = off");// Disable photon channel (much slower)
	}
	//---------------------------------------------------------------
	// e+ e- -> z -> t ttbar
	else if (process == 1){
		// Just turn on the weak single boson generation.
		pythia.readString("WeakSingleBoson:ffbar2ffbar(s:gm) = on");
		pythia.readString("23:oneChannel = 1 1. 0 6");// Set Z (23) to just b b-bar decay products. Bright-Wigner tail of Z dominates at mass > ~91 GeV.
		pythia.readString("22:onMode = off");// Disable photon channel (much slower)

	}
	//---------------------------------------------------------------
	// e+ e- -> z -> w+ w-
	else if (process == 2){
		// enable process
        pythia.readString("999999:addChannel = 1 1. 101 24 -24");//ww
   		//pythia.readString("WeakDoubleBoson:ffbar2WW = on");
		//pythia.readString("WeakZ0:gmZmode = 1");
     
	}
	//---------------------------------------------------------------
	// e+ e- -> g g
	else if (process == 3){
		// enable process

	}
	//---------------------------------------------------------------
	// e+ e- -> h z
	else if (process == 4){
		pythia.readString("HiggsSM:ffbar2HZ = on");
	}


	// Limit decay length to be within 5fm
	pythia.readString("ParticleDecays:limitTau0 = on");
	pythia.readString("ParticleDecays:tau0Max = 2.e-12"); // in mm/c


	// Initialize pythia
	pythia.init();

	// record run time
	time_t start;
	start = time(NULL);

	// Begin event loop. Generate event. Skip if error. List first one.
	for (int iEvent = 0; iEvent < numEvents; ++iEvent){
      if (iEvent%25000==0)
        cout<< iEvent << " events" <<  endl;
	  if (!pythia.next()) continue; // Did event succeed?
	  analyzeEvent(CMS, pythia.event, seed);

	}// End Master Event Loop

	//--------------------------------------------------------------------------
	// Performance Statistics
	//--------------------------------------------------------------------------
	time_t end; //
	//pythia.stat();
	end = time(NULL);
	int diff = difftime (end,start);
	cout << "Time Elapsed for " << numEvents << " events: " << diff <<"s" << endl ;
	cout << "Generated " << antideuteron << " antideuterons (coalescence)" << endl ;
	cout << "Generated " << antideuteron_cross_sec << " antideuterons (cross_sec)" << endl ;
	//--------------------------------------------------------------------------

	return;
}


//------------------------------------------------------------------------------
// Computes the absolute distance in invariant momentum space of a 4 vector
//------------------------------------------------------------------------------
double computeDistance(Vec4 v1, Vec4 v2){
	double p0 = v1.e()  - v2.e(); 
	double px = v1.px() - v2.px();
	double py = v1.py() - v2.py();
	double pz = v1.pz() - v2.pz(); 
	return sqrt(-p0*p0 + px*px + py*py + pz*pz );
}


// Just return distance in invariant momentum space
float dist2(Vec4 p[]){
	// Find Centroid
    double d0 = p[1].e() -p[0].e(); 
    double dx = p[1].px()-p[0].px();
    double dy = p[1].py()-p[0].py();
    double dz = p[1].pz()-p[0].pz();
    return sqrt(-d0*d0 + (dx*dx + dy*dy + dz*dz));
}// end dist2


// Faster algorithm for 3 points
double getMinBoundCirc3(Vec4 pp[]){
	// Compute all three sides of the triangle	
	double a = computeDistance(pp[0],pp[1]);
	double b = computeDistance(pp[0],pp[2]);
	double c = computeDistance(pp[1],pp[2]);

        // Determine if triangle is obtuse, or acute (or right)
        if (a >= b && a >= c){ // If r01 is largest
                // If acute or right, use circumscribed circle radius
                if (b*b+c*c >= a*a){return a*b*c/sqrt( (a+b+c)*(b+c-a)*(c+a-b)*(a+b-c) );}
                else {return a/2.;} // else return the half distance between the farthest two 
        }
        
        else if (b >= a && b >= c){
                if (a*a+c*c >= b*b){return a*b*c/sqrt( (a+b+c)*(b+c-a)*(c+a-b)*(a+b-c) );}
                else {return b/2.;} // else return the half distance between the farthest two 
        }
        else{
                if (b*b+a*a >= c*c){return a*b*c/sqrt( (a+b+c)*(b+c-a)*(c+a-b)*(a+b-c) );}
                else {return c/2.;} // else return the half distance between the farthest two 
        }
        
        // If the above fails, then we must return the radius of the circumcircle
        //return 0.;
        //return a*b*c/sqrt( (a+b+c)*(b+c-a)*(c+a-b)*(a+b-c) );
}



//--------------------------------------------------------------
// This section deals with the position space cuts
//--------------------------------------------------------------

// Return position space distance squared
double posDist2(Vec4 p1, Vec4 p2){
    return (p2.px()-p1.px())*(p2.px()-p1.px())+(p2.py()-p1.py())*(p2.py()-p1.py())+(p2.pz()-p1.pz())*(p2.pz()-p1.pz()); 
}

// Check that three particles are within 2fm of eachother when coalescing.
bool checkPos(Vec4 pp[],int numParts){
    double d_cut = 2e-12; // 2 fm in units of mm
	if (numParts == 2){
	    if (posDist2(pp[0], pp[1]) < d_cut*d_cut){return true;}
	}
	else if (numParts==3){
	    if ((posDist2(pp[0], pp[1]) < d_cut*d_cut) &&
    	    (posDist2(pp[0], pp[2]) < d_cut*d_cut) &&
    	    (posDist2(pp[1], pp[2]) < d_cut*d_cut)){
    	    return true;
    	}
	}
    return false;
}
        

// Returns the centroid of 2 particles
Vec4 getCentroid(Vec4 p1, Vec4 p2){
        Vec4 centroid;
        centroid.p(0,0,0,0);
        // Find Centroid
        centroid.operator +=(p1);
        centroid.operator +=(p2);
	centroid.rescale3(.5);
	return centroid;
}        


//----------------------------------------------------------------------------
// writeEvent.  Write an event property to file
// 	Input:
//		-CMS: CMS Energy of the pythia simulation (i.e.)
// 		-numParticles: number of particles passed
//		-part: A reference to the array of particles
//      -pCoal: distance required for coalescence
//----------------------------------------------------------------------------
void writeEvent(double CMS, int numParticles, Particle part[],float pCoal, int method, float weight){
	Vec4 total;
	total.p(0,0,0,0);
	int A = 0;
	int Z = 0;

	for (int i=0; i< numParticles; i++){
		// Energy
		total.operator +=(part[i].p());
		// Charge and Mass
		if (part[i].id()==PDG_pbar) {Z-=1; A-=1;}
		else if (part[i].id() == PDG_nbar) {A-=1;} // Also give negative atomic number to anti-particles to distinguish nbar from n
		else if (part[i].id()==-PDG_pbar) {Z+=1; A+=1;} // Proton
		else if (part[i].id() == -PDG_nbar) {A+=1;}     // Neutron
	}

	
	// Write to file  (CMS, A, Z, Particle Energy)
	if (numParticles == 1){
		ScopedLock lck(lock_pbar); // locks the mutex
		//dbar_eventFile << CMS << " " << A << " " << Z << " " << total.e() << " " << pCoal << "\n";
		pbar_eventFile << CMS << " " << A << " " << Z << " " << total.e()-.938 << " " << pCoal << " "<< method <<" "<< weight << "\n";
	}    

	// Write to file  (CMS, A, Z, Particle Energy)
	if (numParticles == 2){
		ScopedLock lck(lock_dbar); // locks the mutex
		//dbar_eventFile << CMS << " " << A << " " << Z << " " << total.pAbs2()/(2*1.8765) << " " << pCoal << "\n";
		dbar_eventFile << CMS << " " << A << " " << Z << " " << total.e()-2*.938 << " " << pCoal << " "<< method <<" "<< weight<< "\n";
        //cout << total.pAbs2()/(2.*1.8765) << " " <<  << endl;
	}
	else if (numParticles > 2){
		ScopedLock lck(lock_heavy); // locks the mutex // openmp
		//heavy_eventFile << CMS << " " << A << " " << Z << " " << total.pAbs2()/(2*2.815) << " " << pCoal << "\n";
		heavy_eventFile << CMS << " " << A << " " << Z << " " << total.e()-3*.938 << " " << pCoal << " "<< method << " "<< weight<< "\n";
	}
}


void writeEvent_dbar(double CMS, int numParticles, double E,float pCoal, int method, float weight){
    Vec4 total;
    total.p(0,0,0,0);
    int A = -2;
    int Z = -1;
    
    // Write to file  (CMS, A, Z, Particle Energy)
    if (numParticles == 2){
        ScopedLock lck(lock_dbar); // locks the mutex
        //dbar_eventFile << CMS << " " << A << " " << Z << " " << total.pAbs2()/(2*1.8765) << " " << pCoal << "\n";
        dbar_eventFile << CMS << " " << A << " " << Z << " " << E-2*.938 << " " << pCoal << " "<< method <<" "<< weight<< "\n";
        //cout << total.pAbs2()/(2.*1.8765) << " " <<  << endl;
    }
}


//----------------------------------------------------------------------------
// writeAngle.  Write the cosine of the angle between 2 momentum vectors
// 	Input:
//		-CMS: CMS Energy of the pythia simulation (i.e.)
//		-part: A reference to the array of particles
//      -pCoal: distance required for coalescence
//----------------------------------------------------------------------------
void writeAngle(double CMS, Particle part[],float pCoal){
	Vec4 p0 = part[0].p();
	Vec4 p1 = part[1].p();
	// Compute Angle
	double cosTheta = dot3(p0,p1)/sqrt(dot3(p1,p1)*dot3(p0,p0));
	// Write to file  (CMS, A, Z, Particle Energy)
	ScopedLock lck(lock_dbar); // locks the mutex
	dbar_eventFile << CMS << " " << cosTheta << " " << pCoal << "\n";
}


void writeWW(double CMS, Particle part){
    ScopedLock lck(lock_ww); // locks the mutex
	// Write to file  (CMS, A, Z, Particle Energy)
	ww_eventFile << CMS << " " <<   part.e() << " " << "\n";
	//ww_eventFile << CMS << " " <<   part.pAbs2()/(2.*80.4) << " " << "\n";
}




/////////////////////////////////////////////////////////////////
// Event by event analysis:
// Input
// 		-event: the event to be analyzed
/////////////////////////////////////////////////////////////////
void analyzeEvent(double CMS, Event event, int seed){
	// Store antinucleon lists
	int pbarList [100];     for (int i=0; i<100; i++) pbarList[i] = -1;
	int antiNucIndex = 0;

	srand (time(NULL)*seed); // thread safe init for Pseudo RNG

    // Store nucleon lists
	//int pList [100];     for (int i=0; i<100; i++) pList[i] = -1;
	//int NucIndex = 0;

	for (int i = 0; i < event.size(); ++i){
		Particle& part = event[i];

		if (part.isFinal() && (part.id() == PDG_pbar || part.id() == PDG_nbar)){		
			pbarList[antiNucIndex] = i;
			antiNucIndex +=1;
			//cout << " pbar: "<< pythia.event[i].p();
		}// nucleon test


		/*else if (part.isFinal() && (part.id() == -PDG_pbar || part.id() == -PDG_nbar)){
			pList[NucIndex] = i;
			NucIndex +=1;
		}// nucleon test*/
		
		/*else if (part.status() == -22 && (part.id()== 24 || part.id() == -24))
		{
		    writeWW(CMS,part);
		}*/
		
	} // particle loop


	if (antiNucIndex > 0)
	{
	  // loop over all antinucleon pairs (upper-triangle only to avoid double analysis)
	  for (int i = 0; i<antiNucIndex ; i++){
		  Particle& part1 = event[pbarList[i]];

		  Particle partArray[3] = {part1,0,0};  // Output anti-protons and anti-neutrons
		  
          //writeEvent(CMS, 1, partArray,(float)(0.),0, 1);

		  for (int j = i+1; j< antiNucIndex ; j++){
			  Particle& part2 = event[pbarList[j]];
			  

			  ///////////////////////////////////////////////////////
			  // Check for Antidueterons
			  Vec4 pVecs[4] = {part1.p(), part2.p(),0,0};
    		  Vec4 xVecs[4] = {part1.vProd(), part2.vProd(),0,0};
			  float dist = dist2(pVecs);

		      partArray[1] = part2;
			  // Write angular information to file.
			  //writeAngle(CMS, partArray,dist);
				
			  // Check coalesence condition for antideuterons
			  if (dist <= pCoalHigh){
				//if (checkPos(xVecs,2)==true)
			    //{
				    antideuteron +=1 ;
				    // Write event to file
				    writeEvent(CMS, 2, partArray,dist, 0, 1);
				//}
			  }

			  // If use Dal & Raklev rather than 
			  if (cross_sec_method == true){
			  		// Compute probabilties for each process in table I of Dal & Raklev
			  		double prob[8];

			  		// if (    ((part1.id() == PDG_pbar) && (part2.id() == PDG_nbar)) 
			  		// 	 || ((part2.id() == PDG_pbar) && (part1.id() == PDG_nbar)) ) {
				  	// 	// pn processes
				  	// 	prob[0] = eq7(dist)*sigma_0; // can be larger than 1, but don't care.
				  	// 	prob[1] = 0.5*eq10(dist)*sigma_0;
				  	// 	prob[2] = (2*eq13_pi0pi0(dist)+0.5*eq13_pi0piplus(dist))*sigma_0;
				  	// 	prob[3] = eq13_pi0pi0(dist)*sigma_0;
				  	// }

			  		// else if ((part1.id() == PDG_pbar) && (part2.id() == PDG_pbar)) {
			  		// 	// pp -> d pi^- 
			  		// 	prob[4] = eq10(dist)*sigma_0;
			  		// 	prob[5] = eq13_pi0piplus(dist)*sigma_0;
			  		// }

			  		// else if ((part1.id() == PDG_nbar) && (part2.id() == PDG_nbar)) {
			  		// 	prob[6] = eq10(dist)*sigma_0;
			  		// 	prob[7] = eq13_pi0piplus(dist)*sigma_0;
			  		// }

                    if (    ((part1.id() == PDG_pbar) && (part2.id() == PDG_nbar)) 
                      || ((part2.id() == PDG_pbar) && (part1.id() == PDG_nbar)) ) {
                     // pn processes
                     prob[0] = xs_pn_dgamma(dist)*sigma_0; // can be larger than 1, but don't care.
                     prob[1] = xs_pn_dpi0(dist)*sigma_0;
                     prob[2] = xs_pn_dpippim(dist)*sigma_0;
                     prob[3] = xs_pn_dpi0pi0(dist)*sigma_0;
                    }

                    else if ((part1.id() == PDG_pbar) && (part2.id() == PDG_pbar)) {
                     // pp -> d pi^- 
                     prob[4] = xs_pp_dpip(dist)*sigma_0;
                     prob[5] = xs_pp_dpippi0(dist)*sigma_0;
                    }

                    else if ((part1.id() == PDG_nbar) && (part2.id() == PDG_nbar)) {
                     prob[6] = xs_nn_dpim(dist)*sigma_0;
                     prob[7] = xs_nn_dpimpi0(dist)*sigma_0;
                    }

			  		// Negelecting formation of two dbars from the same pair of nucleons
			  		// since this is extremely rare. 

			  		// We also implement the nested sampling of Dal & Raklev.
			  		for (int sample=0; sample<5; sample++){
				  		for (int i_P=0; i_P<8; i_P++){
				  			//cout<< "prob" << i << " " << prob[i] << endl;
				  			// random number between 0,1
				  			double r = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1));
				  			// if we succesfully created a dbar, sample the 2 or 3 body phase space 
				  			if (r<prob[i_P]){	
                                //--------------------------------------------------
                                // if 2-body phase space
                                vec4 p_dbar_lab; 
                                if ((i_P==0) || (i_P==1) || (i_P==4) || (i_P==6)){
                                    double E = 2*sqrt(mp*mp+0.25*dist*dist); // COM energy 
                                    double m_2 = 0; // mass of second particle
                                    if (i_P!=0){
                                        m_2=mpic; 
                                    }
                                    // Now two body phase space (see e.g. http://www-pnp.physics.ox.ac.uk/~libby/Teaching/Lecture4.pdf)
                                    double p_dbar = sqrt( (E*E-pow(2*.938-m_2,2))*(E*E-pow(2*.938+m_2,2)) )/ (2*E);

                                    // Emit dbar in random direction with correct momentum 
                                    vec4 p_dbar_4vec = vec4(sqrt( (2*.938)*(2*.938) + p_dbar*p_dbar),  randOnSphere()*p_dbar );
                                    // boost to galactic frame 
                                    // Vec4 p_lab = part1.p()+part2.p(); // pythia Vec4
                                    // vec4(p_lab.e(), p_lab.px(), p_lab.py(), p_lab.pz() );// manually convert to vec4 for lorentz transformation
                                    vec4 p1 = vec4(part1.e(),part1.px(),part1.py(),part1.pz());
                                    vec4 p2 = vec4(part2.e(),part2.px(),part2.py(),part2.pz());
                                    p_dbar_lab = boostToLabFrame( p1, p2, p_dbar_4vec); 
                                }//end if 2-body

                                //--------------------------------------------------
                                // if 3-body phase space
                                else{
                                    vec4 p1 = vec4(part1.e(),part1.px(),part1.py(),part1.pz());
                                    vec4 p2 = vec4(part2.e(),part2.px(),part2.py(),part2.pz());
                                    // Calculate Energy of the third particle in the COM frame
                                    double e = dalitz(calc_s(p1,p2), mpic, mpic, 0.938);
                                    // 4 momentum of dbar in COM frame 
                                    double p_dbar = sqrt(e*e-pow(2*.938,2));
                                    vec4 p_dbar_4vec = vec4(e,  randOnSphere()*p_dbar);
                                    p_dbar_lab = boostToLabFrame( p1, p2, p_dbar_4vec); 
                                }
                                
				  				// the 1 is the cross-section method vs coal 
				  				writeEvent_dbar(CMS, 2, p_dbar_lab[0], dist, 1, .2);	
				  				antideuteron_cross_sec +=.2 ;
				  			}
				  		}
			  		}


			  	}// end if cross_section

			  ///////////////////////////////////////////////////////////////////
			  // Check for Antihelium 3 or Tritium
			  for (int k = j + 1; k < antiNucIndex; k++){
				  Particle& part3 = event[pbarList[k]];
				  pVecs[2] = part3.p();

				  // Check coalesence condition for antihelium 3
                  dist = 2.*getMinBoundCirc3(pVecs);
				  
				  //cout<<dist << " " << dist2 << endl;
				  
				  if (dist > pCoalHigh) continue;
				  else{
				    xVecs[2] = part3.vProd();
//				    if (checkPos(xVecs,3)==true)
//				    {
					    antihelium3 +=1 ;
					    partArray[2] = part3;
					    writeEvent(CMS, 3, partArray,dist, 0, 1);
//					}
				  }

				  /*///////////////////////////////////////////////////////////////
				  // Check for Antihelium 4
				  for (int l = k + 1 ; l < antiNucIndex; l++){
					  Particle& part4 = event[pbarList[k]];
					  pVecs[3] = part4.p();
					  // Check coalesence condition for antihelium 4
					  if (checkCoal(4, pVecs) == true){
						  antihelium4 +=1 ;
						  //cout << "Antihelium 4!!!" << endl;
						  Particle partArray[4] = {part1, part2, part3, part4};
						  writeEvent(CMS, 4, partArray);
					  }
				  }// end l*/
			  }// end k
		  }// end j
	  }// end i
	}// end test

	//----------------------------------------------------------
	// Loop over nucleons
	/*if (NucIndex > 0)
		{
		  // loop over all antinucleon pairs (upper-triangle only to avoid double analysis)
		  for (int i = 0; i<NucIndex ; i++){
			  Particle& part1 = event[pbarList[i]];

			  Particle partArray[1] = {part1};  // Output anti-protons and anti-neutrons
			  //writeEvent(CMS, 1, partArray);

			  for (int j = i+1; j< NucIndex ; j++){
				  Particle& part2 = event[pbarList[j]];

				  ///////////////////////////////////////////////////////
				  // Check for Antidueterons
				  Vec4 pVecs[4] = {part1.p(), part2.p(),0,0};

					  float dist = dist2(pVecs);
					  // Check coalesence condition for antideuterons
					  if (dist <= pCoalHigh){
						antideuteron +=1 ;
						// Write event to file
						Particle partArray[2] = {part1, part2};
						writeEvent(CMS, 2, partArray,dist);
					  }



				  ///////////////////////////////////////////////////////////////////
				  // Check for Antihelium 3 or Tritium
				  for (int k = j + 1; k < NucIndex; k++){
					  Particle& part3 = event[pbarList[k]];
					  pVecs[2] = part3.p();

					  // Check coalesence condition for antihelium 3  ***DIAMETER NOT RADIUS***
					  //float dist = 2*getMinBoundCirc(pVecs, 3);
					  float dist = 2.*getMinBoundCirc3(pVecs);
					  if (dist > pCoalHigh) continue;
					  else{
						antihelium3 +=1 ;
						//cout << "Antihelium 3!!!" << endl;
						Particle partArray[3] = {part1, part2, part3};
						writeEvent(CMS, 3, partArray,dist);
					  }
				  }// end k
			  }// end j
		  }// end i
		}*/

} // end analyzeEvent










