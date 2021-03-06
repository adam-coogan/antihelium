//-------------------------------------------------------------------------
// Run pythia for several different DM annihilation processes and check
// check formation rates of anti-nuclei up to A=4
//
// Author: Eric Carlson (originally), Adam Coogan
//-------------------------------------------------------------------------
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <ctime>
#include <math.h>
#include <stdio.h>
#include <boost/thread.hpp>
#include "Pythia8/Pythia.h"
#include <fstream>

using namespace Pythia8;
using namespace std;


//------------------------------------------------------------------------
// Function Prototypes
//------------------------------------------------------------------------
double computeDistance(Vec4 v1, Vec4 v2);
bool checkCoal(double pDist);
double mbsRadius(int numParticles, Vec4 p[]);
void writeEvent(double CMS, int numParticles, Particle part[]);
int main(int argc, char *argv[]);
void analyzeEvent(double CMS, Event event);
void pythiaThread(int numEvents, double CMS, int seed, int process);

//------------------------------------------------------------------------
// Global Declarations
//------------------------------------------------------------------------
double pCoal = 0.5
int antideuteron = 0; // Number of antideuterons
int antihelium3  = 0;
int antihelium4  = 0;

//------------------------------------------------------------------------
// Some PDG Particle Codes
const int PDG_pbar = -2212; // Antiproton
const int PDG_nbar = -2112; // Antineutron
const int PDG_e    = 11; // electron
const int PDG_ebar = -11; // electron
const int PDG_b    = 5;
const int PDG_bbar = -5;

// Filewriter
ofstream eventFile;

/////////////////////////////////////////////////////////////////////////
// main() starts each thread.  Uses boost libraries for multi-threading
/////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {

	if (argc != 4){
		cout << "\nSYNTAX:\nGenAntimatter <Event File> <Process> <p_coal (GeV)>" << endl;
		cout << "Processes:"
                "\n0: e+ e- -> z -> b bbar"
                "\n1: e+ e- -> z -> t tbar"
                "\n2: e+ e- -> z -> w+ w-"
                "\n3: e+ e- -> g g (NOT IMPLEMENTED YET)"
                "\n4: e+ e- -> h z "<< endl;
		return 0;
	}

	//			0: e+ e- -> z -> b bbar
	//			1: e+ e- -> z -> t tbar
	// 			2: e+ e- -> z -> w+ w-
	//			3: e+ e- -> g g // NOT IMPLEMENTED
	//			4: e+ e- -> h z
	int process = atoi(argv[2]);
	pCoal = atof(argv[3]);
	int numEvents = (int) (1e5); // Total number of events.  Will be distributed over threads.

	// Find number of CPUs
	//long numCPU = sysconf( _SC_NPROCESSORS_ONLN );
	const long numCPU = 1;  // Uncomment to specify num CPUs
	cout << "Using " << numCPU << " CPUs..." << endl;
	int seed;

	// Initialize file writer
	eventFile.open(argv[1]);
	eventFile << "# RUNDETAILS: time = " << time(NULL) << ", numEvents = " << numEvents
        << ", process number = " << process << endl;
    eventFile << "# Columns: E_CM, A (mass number Z+N), Z (atomic/proton number), particle energy (GeV), "
              "p_coal (GeV)" << endl;

	//int numMasses = 3;
	double CMS[4] = {20,200,1000,2000};
	int numMasses = 4;

	for (int massidx=0; massidx < numMasses; massidx++){
		// Thread Array
		boost::thread threads[numCPU];

		// Reinit the RNG
		//srand(time(NULL)*atoi(argv[3]));
		srand(time(NULL));

		// Start each thread
		for (int i = 0 ; i < numCPU ; i++){
			seed = (int) (abs(900000000*rand())); // this 9e8 is max seed to pass to RNG
			threads[i] = boost::thread(pythiaThread, (int) (numEvents/numCPU), CMS[massidx], seed,process);
		}

		// Join threads
		for (int i = 0 ; i < numCPU ; i++) threads[i].join();
	}
	// Close filewriter
	eventFile.close();
	cout << "Job finished!" << endl;
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
    std::cout << "Setting up pythia" << std::endl;
	Pythia pythia;
    std::cout << "Done setting up pythia" << std::endl;

	// Output Message every 10,000 events
	pythia.readString("Next:numberCount = 10000.");

	// Electron-Positron Collisions
	std::stringstream s;
	s << "Beams:eCM = " << CMS;
	pythia.readString(s.str()); // CMS Energy in GeV
	pythia.readString("Beams:idA = 11");
	pythia.readString("Beams:idB = -11");

	// Init Random number generator
	pythia.rndm.init(seed); // Generate random seed for pseudo-random generator);

	//---------------------------------------------------------------
	// PROCESS SETTINGS
	//---------------------------------------------------------------
	//---------------------------------------------------------------
	// e+ e- -> z -> b bbar
	if (process == 0){
		// Just turn on the weak single boson generation.
		pythia.readString("WeakSingleBoson:ffbar2ffbar(s:gm) = on");
		// Setup channels
		pythia.readString("23:oneChannel = 1 1. 0 5");// Set Z (23) to just b b-bar decay products. Bright-Wigner tail of Z dominates at mass > ~91 GeV.
		pythia.readString("22:onMode = off");// Disable photon channel (much slower)
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
		pythia.readString("WeakDoubleBoson:ffbar2WW = on");
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


	// Limit decay length to be within 50fm
	//pythia.readString("ParticleDecays:limitTau0 = on");
	//pythia.readString("ParticleDecays:tau0Max = 5.e-11"); // in mm/c


	// Initialize pythia
	pythia.init();

	// record run time
	time_t start;
	start = time(NULL);

	// Begin event loop. Generate event. Skip if error. List first one.
	for (int iEvent = 0; iEvent < numEvents; ++iEvent){

	  if (!pythia.next()) continue; // Did event succeed?
	  analyzeEvent(CMS, pythia.event);

	}// End Master Event Loop

	//--------------------------------------------------------------------------
	// Performance Statistics
	//--------------------------------------------------------------------------
	time_t end; //
	//pythia.stat();
	end = time(NULL);
	int diff = difftime (end,start);
	cout << "Time Elapsed for " << numEvents << " events: " << diff <<"s" << endl ;
	cout << "Generated " << antideuteron << " antideuterons." << endl ;
	//--------------------------------------------------------------------------

	return;
}

//------------------------------------------------------------------------------
// Computes the absolute distance of the first 3 arguments of a 4 vector
//------------------------------------------------------------------------------
double computeDistance(Vec4 v1, Vec4 v2){
	double px = v1.px() - v2.px();
	double py = v1.py() - v2.py();
	double pz = v1.pz() - v2.pz();
	double pdiff = pow( px*px + py*py + pz*pz , .5 );
	return pdiff;
}

//------------------------------------------------------------------------------
// Check the coalescence condition.
//------------------------------------------------------------------------------
bool checkCoal(double pDist){
    // Check whether most distant point is in coalescence radius
    if (pDist > pCoal / 2) {
        return false;
    } else {
        return true;
    }
}// end checkCoal

// TODO: this clearly breaks down for n > 3, but that's not a big problem.
double mbsRadius(int numParticles, Vec4 p[]){
	Vec4 centroid;
	centroid.p(0,0,0,0);

	// Find Centroid
	for (int i=0; i< numParticles; i++) {
        centroid.operator +=(p[i]);
    }

	centroid.rescale3(1./((double) numParticles));

	// Find the particle furthest from the centroid
    double maxDist = 0;
	for (int i=0; i<numParticles; i++){
		maxDist = max(computeDistance(centroid, p[i]), maxDist);
	}

	return maxDist;
}

//----------------------------------------------------------------------------
// writeEvent.  Write an event property to file
// 	Input:
//		-CMS: CMS Energy of the pythia simulation (i.e.)
// 		-numParticles: number of particles passed
//		-part: A reference to the array of particles
//----------------------------------------------------------------------------
void writeEvent(double CMS, int numParticles, Particle part[]){
	Vec4 total;
	total.p(0,0,0,0);
	int A = 0;
	int Z = 0;

	for (int i=0; i< numParticles; i++){
		// Energy
		total.operator +=(part[i].p());
		// Charge and Mass
		if (part[i].id() == PDG_pbar) {
            Z-=1;
            A+=1;
        } else if (part[i].id() == PDG_nbar) {
            A+=1;
        } else if (part[i].id() == -PDG_pbar) { // Count this as a pbar since the process conserves B?
            Z-=1;
            A+=1;
        } else if (part[i].id() == -PDG_nbar) { // Count this as an nbar since the process conserves B?
            A+=1;
        }
	}
	//cout<< total.e() << endl;

	// Write to file  (CMS, A, Z, Particle Energy, p_coal)
	eventFile << CMS << " " << A << " " << Z << " " << total.e() << " " << pCoal << "\n";
}

/////////////////////////////////////////////////////////////////
// Event by event analysis:
// Input
// 		-event: the event to be analyzed
/////////////////////////////////////////////////////////////////
void analyzeEvent(double CMS, Event event){
	// Store antinucleon lists
	int pbarList [100];     for (int i=0; i<100; i++) pbarList[i] = -1;
	int pList [100];     for (int i=0; i<100; i++) pList[i] = -1;
	int antiNucIndex = 0;
	int NucIndex = 0;

	for (int i = 0; i < event.size(); ++i){
		Particle& part = event[i];

		if (part.isFinal() && (part.id() == PDG_pbar || part.id() == PDG_nbar)){
			pbarList[antiNucIndex] = i;
			antiNucIndex +=1;
			//cout << " pbar: "<< pythia.event[i].p();
		}// nucleon test

		else if (part.isFinal() && (part.id() == -PDG_pbar || part.id() == -PDG_nbar)){
			pList[NucIndex] = i;
			NucIndex +=1;
		}// nucleon test

	} // particle loop
	
	if (NucIndex > 0)
	{
	  for (int i = 0; i<NucIndex ; i++){
		  Particle& part1 = event[pbarList[i]];

		  Particle partArray[1] = {part1};  // Output protons and neutrons
		  writeEvent(CMS, 1, partArray);
	  }
	}


	if (antiNucIndex > 0)
	{
	  // loop over all antinucleon pairs (upper-triangle only to avoid double analysis)
	  for (int i = 0; i<antiNucIndex ; i++){
		  Particle& part1 = event[pbarList[i]];

		  Particle partArray[1] = {part1};  // Output anti-protons and anti-neutrons
		  writeEvent(CMS, 1, partArray);

		  for (int j = i+1; j< antiNucIndex ; j++){
			  Particle& part2 = event[pbarList[j]];

			  ///////////////////////////////////////////////////////
			  // Check for antideuterons
			  Vec4 pVecs[4] = {part1.p(), part2.p(),0,0};

			  // Check coalesence condition for antideuterons
              double pDist2 = mbsRadius(2, pVecs);

			  if (checkCoal(pDist2) == true){
				antideuteron +=1 ;
				cout << "Antideuteron!!!" << endl;

				// Write event to file
				Particle partArray[2] = {part1, part2};
				writeEvent(CMS, 2, partArray);
			  } else {
                  continue; // Don't check for antihelium if the first two don't coalesce!
              }

			  ///////////////////////////////////////////////////////////////////
			  // Check for Antihelium 3 or Tritium
			  for (int k = j + 1; k < antiNucIndex; k++){
				  Particle& part3 = event[pbarList[k]];
				  pVecs[2] = part3.p();

				  // Check coalesence condition for antihelium 3
                  double pDist3 = mbsRadius(3, pVecs);

				  if (checkCoal(pDist3) == false) {
                      continue;
                  } else {
				  	antihelium3 +=1 ;
				  	cout << "Antihelium 3!!!" << endl;
				  	Particle partArray[3] = {part1, part2, part3};
				  	writeEvent(CMS, 3, partArray);
				  }

				  ////////////////////////////////////////////////////////////////

				  // Check for Antihelium 4
				  for (int l = k + 1 ; l < antiNucIndex; l++){
					  Particle& part4 = event[pbarList[k]];
					  pVecs[3] = part4.p();

					  // Check coalesence condition for antihelium 4
                      double pDist4 = mbsRadius(3, pVecs);

					  if (checkCoal(pDist4) == true){
						  antihelium4 +=1 ;
						  cout << "Antihelium 4!!!" << endl;
						  Particle partArray[4] = {part1, part2, part3, part4};
						  writeEvent(CMS, 4, partArray);
					  }
				  }// end l
			  }// end k
		  }// end j
	  }// end i
	}// end test
} // end analyzeEvent


