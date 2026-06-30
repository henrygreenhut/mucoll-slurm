#Basic Scripts for Getting Started, used with Submit_basic.sh

#Imports
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

#Getting your Root File (replace with the actual name)
file = uproot.open(f"XXXXX.root")
#Getting the main tree needed
events = file["events"]

'''
There are 4 main types of objects or things you can look at from the root files: Hits, Clusters, PFOs (Particle Flow Objects), and Tracks (only from charged particles)
We will only look at hits in one case here.
This will just be some brief plotting scripts that look into the types of objects, and some of the information avaliable.
Depending on if you have tracking turned on or not, you may have PFOS that look weird (not being able to recognize the correct particle pid (particle type))
If you ever don't know what you have access to, you can simply write print(event.keys()) after introducting events and it should print avaliable variables.
Warning: Not all of these things are interesting to study (for example looking at the phi distribution of all random clusters, as opposed to a leading clusters, has proved to not be interesting)

'''
##################################################################
#1: Multiplicity  (how many of X things exist in an event)
#To get multiplicity, you can just see how long an array is that holds some per object characteristic for the object type


#Before looping through events you should obtain all the arrays outside of the loop, if not it will take a long time 
''' PFOS '''
PandoraPFOEnergy = events["PandoraPFOs/PandoraPFOs.energy"].array()
PandoraPFOs = events["PandoraPFOs"] 
PandoraPFOEnergy2 = PandoraPFOs["PandoraPFOs.energy"].array() #This is another way to write your variables by decomposing them at '/' signs
#PandoraPFOEnergy and PandoraPFOEnergy2 should be equivalent

''' Clusters '''
clusters = events["PandoraClusters"]
ClusterEnergy   = clusters["PandoraClusters.energy"].array() #Your final variable you're actually using should have a '.array()' attached to it

''' Tracks '''
deduped_tracks = events["DedupedTracks_objIdx"]
index_deduped = deduped_tracks["DedupedTracks_objIdx.index"].array()
track_states = events["_AllTracks_trackStates"]
#I am not sure what the accepted way to get tracks is nowadays, but this will get you a number that could refer to the number of tracks in different parts of the detector

pfos_count = []
clus_count = []
tracks_count = []

for i in range((events.num_entries)): #you cannnot just write len(events), will not give the full amount of events
    num_pfo = len(PandoraPFOEnergy[i])
    num_clus = len(ClusterEnergy[i])
    num_tracks = len(index_deduped[i])
    pfos_count.append(num_pfo)
    clus_count.append(num_clus)
    tracks_count.append(num_tracks)

pfos_count = np.array(pfos_count)
clus_count = np.array(clus_count)
tracks_count = np.array(tracks_count)

bins = 10
plt.hist(tracks_count, bins=bins, edgecolor='black')
plt.xlabel("Number of Tracks per Event")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Number of Tracks per Event")
plt.tight_layout()
plt.savefig("basicplot_trackmultiplicity.pdf")
plt.close()

bins = 20
plt.hist(clus_count, bins=bins, edgecolor='black')
plt.xlabel("Number of Clusters per Event")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Number of Clusters per Event")
plt.tight_layout()
plt.savefig("basicplot_clustermultiplicity.pdf")
plt.close()

bins = 10
plt.hist(pfos_count, bins=bins, edgecolor='black')
plt.xlabel("Number of PFOs per Event")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Number of PFOs per Event")
plt.tight_layout()
plt.savefig("basicplot_pfomultiplicity.pdf")
plt.close()

#################################################################
#1.5: Bonus: Number of Hits in an Event
#This is its own category since it needs different loops

hits_begin_all = clusters["PandoraClusters.hits_begin"].array()
hits_end_all   = clusters["PandoraClusters.hits_end"].array()
hits_per_event = []
for i in range((events.num_entries)):
    hits = 0
    hits_begin_arr = hits_begin_all[i]
    hits_end_arr   = hits_end_all[i]
    for j in range(len(hits_begin_arr)):
        hits += (hits_end_all[i][j] - hits_begin_all[i][j])
    hits_per_event.append(hits)



bins = 50
plt.hist(hits_per_event, bins=bins, edgecolor='black')
plt.xlabel("Number of Hits per Event")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Number of Hits per Event")
plt.tight_layout()
plt.savefig("basicplot_hitmultiplicity.pdf")
plt.close()


##################################################################
#2: Energy  #We'll be looking at all the energy of those items in an event

#From above we already have a variable for PFO Energy, and Cluster Energy
#I don't believe tracks have energy (makes sense)

total_clusenergy_perevent = []
total_pfoenergy_perevent = []
for i in range((events.num_entries)):
    clus_energy = 0
    pfo_energy = 0
    for j in range((len(ClusterEnergy[i]))):
        clus_energy += ClusterEnergy[i][j]
    for j in range((len(PandoraPFOEnergy[i]))):
        pfo_energy += PandoraPFOEnergy[i][j]
    total_clusenergy_perevent.append(clus_energy)
    total_pfoenergy_perevent.append(pfo_energy)

bins = 50
plt.hist(total_clusenergy_perevent, bins=bins, edgecolor='black')
plt.xlabel("Summed Cluster Energy per Event")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Total Cluster Energy")
plt.tight_layout()
plt.savefig("basicplot_clusenergy.pdf")
plt.close()

bins = 50
plt.hist(total_pfoenergy_perevent, bins=bins, edgecolor='black')
plt.xlabel("Summed Pfo Energy per Event")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Total Pfo Energy")
plt.tight_layout()
plt.savefig("basicplot_pfoenergy.pdf")
plt.close()
#################################################################
#3: Phi Distribution: We'll be looking at the phi distribution of all the clusters, and pfos (not event wise)
#Once again a warning that these tracks may not be the accepted tracks currenlty -> especially with the new image, SiRefitter may be dead(?)
#From this you can easily find theta distribution too


## Clusters ##
phi_clusters = clusters["PandoraClusters.phi"].array()

## PFOS ##
#PFOS don't directly have phi, so we're going to build it from momentum
momx_pfos = PandoraPFOs["PandoraPFOs.momentum.x"].array()
momy_pfos = PandoraPFOs["PandoraPFOs.momentum.y"].array()
momz_pfos = PandoraPFOs["PandoraPFOs.momentum.z"].array()

## Tracks ##
phi_tracks = track_states["_AllTracks_trackStates.phi"].array()


total_clusphi = []
total_pfophi = []
total_trackphi = []
for i in range((events.num_entries)):
    for j in range((len(ClusterEnergy[i]))):
        total_clusphi.append(phi_clusters[i][j])
    for j in range((len(PandoraPFOEnergy[i]))):
        phi = np.arctan2(momy_pfos[i][j], momx_pfos[i][j])
        total_pfophi.append(phi)
    if (len(index_deduped[i]) !=0 ):
        indices = index_deduped[i][0] 
        our_tphi = phi_tracks[indices]
        total_trackphi.extend(our_tphi) #because multiple points at once, extend not append

#We need to flatten total_trackphi
bins = 15
plt.hist(total_trackphi, bins=bins, edgecolor='black')
plt.xlabel("Phi")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Phi for Meaningful Tracks [0]")
plt.tight_layout()
plt.savefig("basicplot_trackphi_0.pdf")
plt.close()

bins = 15
plt.hist(total_clusphi, bins=bins, edgecolor='black')
plt.xlabel("Phi")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Phi for All Clusters")
plt.tight_layout()
plt.savefig("basicplot_clusterphi.pdf")
plt.close()

bins = 15
plt.hist(total_pfophi, bins=bins, edgecolor='black')
plt.xlabel("Phi")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Phi for All PFOs")
plt.tight_layout()
plt.savefig("basicplot_pfophi.pdf")
plt.close()




