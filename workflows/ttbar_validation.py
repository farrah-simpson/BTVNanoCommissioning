import coffea
from coffea import hist, processor
import numpy as np
#import awkward1 as ak
import awkward as ak
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask

class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(self):        
        # Define axes
        # Should read axes from NanoAOD config
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cutflow_axis   = hist.Cat("cut",   "Cut")
       
        # Events
        nel_axis   = hist.Bin("nel",   r"N electrons", [0,1,2,3,4,5,6,7,8,9,10])
        nmu_axis   = hist.Bin("nmu",   r"N muons",     [0,1,2,3,4,5,6,7,8,9,10])
        njet_axis  = hist.Bin("njet",  r"N jets",      [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_t_axis = hist.Bin("nbjet_t", r"N tight b-jets",    [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_m_axis = hist.Bin("nbjet_m", r"N medium b-jets",    [0,1,2,3,4,5,6,7,8,9,10])
        nbjet_l_axis = hist.Bin("nbjet_l", r"N loose b-jets",    [0,1,2,3,4,5,6,7,8,9,10])

        # Electron
        el_pt_axis   = hist.Bin("pt",    r"Electron $p_{T}$ [GeV]", 100, 20, 400)
        el_eta_axis  = hist.Bin("eta",   r"$\eta$", 60, -3, 3)
        el_phi_axis  = hist.Bin("phi",   r"$\phi$", 60, -3, 3)
        lelpt_axis   = hist.Bin("lelpt", r"Leading electron $p_{T}$ [GeV]", 100, 20, 200)
        
        # Muons
        mu_pt_axis   = hist.Bin("pt",    r"Muon $p_{T}$ [GeV]", 100, 20, 400)
        mu_eta_axis  = hist.Bin("eta",   r"$\eta$", 60, -3, 3)
        mu_phi_axis  = hist.Bin("phi",   r"$\phi$", 60, -3, 3)
        lmupt_axis   = hist.Bin("lmupt", r"Leading muon $p_{T}$ [GeV]", 100, 20, 200)
        
        # Jet
        jet_pt_axis   = hist.Bin("pt",   r"Jet $p_{T}$ [GeV]", 100, 20, 400)
        jet_eta_axis  = hist.Bin("eta",  r"$\eta$", 60, -3, 3)
        jet_phi_axis  = hist.Bin("phi",  r"$\phi$", 60, -3, 3)
        jet_mass_axis = hist.Bin("mass", r"Jet $m$ [GeV]", 100, 0, 50)
        ljpt_axis     = hist.Bin("ljpt", r"Leading jet $p_{T}$ [GeV]", 100, 20, 400)
        sljpt_axis     = hist.Bin("sljpt", r"Subleading jet $p_{T}$ [GeV]", 100, 20, 400)
 
        # Define similar axes dynamically
        disc_list = ["btagCMVA", "btagCSVV2", 'btagDeepB', 'btagDeepC', 'btagDeepFlavB', 'btagDeepFlavC',]
        btag_axes = []
        for d in disc_list:
            btag_axes.append(hist.Bin(d, d, 50, 0, 1))        
        
        deepcsv_list = ["DeepCSV_trackDecayLenVal_0", "DeepCSV_trackDecayLenVal_1", "DeepCSV_trackDecayLenVal_2", "DeepCSV_trackDecayLenVal_3", "DeepCSV_trackDecayLenVal_4", "DeepCSV_trackDecayLenVal_5", "DeepCSV_trackDeltaR_0", "DeepCSV_trackDeltaR_1", "DeepCSV_trackDeltaR_2", "DeepCSV_trackDeltaR_3", "DeepCSV_trackDeltaR_4", "DeepCSV_trackDeltaR_5"]
        deepcsv_axes = []
        for d in deepcsv_list:
            if "trackDecayLenVal" in d:
                deepcsv_axes.append(hist.Bin(d, d, 50, 0, 2.0))
            else:
                deepcsv_axes.append(hist.Bin(d, d, 50, 0, 0.3))

        # Define histograms from axes
        _hist_jet_dict = {
                'pt'  : hist.Hist("Counts", dataset_axis, jet_pt_axis),
                'eta' : hist.Hist("Counts", dataset_axis, jet_eta_axis),
                'phi' : hist.Hist("Counts", dataset_axis, jet_phi_axis),
                'mass': hist.Hist("Counts", dataset_axis, jet_mass_axis),
            }
        _hist_deepcsv_dict = {
                'pt'  : hist.Hist("Counts", dataset_axis, jet_pt_axis),
                'eta' : hist.Hist("Counts", dataset_axis, jet_eta_axis),
                'phi' : hist.Hist("Counts", dataset_axis, jet_phi_axis),
                'mass': hist.Hist("Counts", dataset_axis, jet_mass_axis),
            }
 
        # Generate some histograms dynamically
        for disc, axis in zip(disc_list, btag_axes):
            _hist_jet_dict[disc] = hist.Hist("Counts", dataset_axis, axis)
        for deepcsv, axis in zip(deepcsv_list, deepcsv_axes):
            _hist_deepcsv_dict[deepcsv] = hist.Hist("Counts", dataset_axis, axis)
        
        _hist_event_dict = {
                'njet'  : hist.Hist("Counts", dataset_axis, njet_axis),
                'nbjet_t' : hist.Hist("Counts", dataset_axis, nbjet_t_axis),
                'nbjet_m' : hist.Hist("Counts", dataset_axis, nbjet_m_axis),
                'nbjet_l' : hist.Hist("Counts", dataset_axis, nbjet_l_axis),
                'nel'   : hist.Hist("Counts", dataset_axis, nel_axis),
                'nmu'   : hist.Hist("Counts", dataset_axis, nmu_axis),
                'lelpt' : hist.Hist("Counts", dataset_axis, lelpt_axis),
                'lmupt' : hist.Hist("Counts", dataset_axis, lmupt_axis),
                'ljpt'  : hist.Hist("Counts", dataset_axis, ljpt_axis),
                'sljpt'  : hist.Hist("Counts", dataset_axis, sljpt_axis),
            }
        
        self.jet_hists = list(_hist_jet_dict.keys())
        self.deepcsv_hists = list(_hist_deepcsv_dict.keys())
        self.event_hists = list(_hist_event_dict.keys())
    
        _hist_dict = {**_hist_jet_dict, **_hist_deepcsv_dict, **_hist_event_dict}
        self._accumulator = processor.dict_accumulator(_hist_dict)
        self._accumulator['sumw'] = processor.defaultdict_accumulator(float)


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")

	selection = PackedSelection()
        weights = Weights(len(events))
        output['sumw'][dataset] += ak.sum(events.genWeight)
        
        ##############
        # Trigger level
        triggers = [
        "HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",    
        ]
        
        trig_arrs = [events.HLT[_trig.strip("HLT_")] for _trig in triggers]
        req_trig = np.zeros(len(events), dtype='bool')
        for t in trig_arrs:
            req_trig = req_trig | t

	selection.add("trigger",trigger)


        ############
        # Event level
        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        goodmuon = ( (events.Muon.pt > 30) & (abs(events.Muon.eta < 2.4)) ) # & (events.Muon.tightId > .5)
        #events.Muon = ak.pad_none(events.Muon, 1, axis=1) 
        #req_muon =(ak.count(events.Muon.pt, axis=1) == 1)
        nmuons = ak.sum(goodmuon,axis=1)

        ## Electron cuts
        # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        nelectrons = ak.sum ( (events.Electron.pt > 30) & (abs(events.Electron.eta) < 2.4), axis = 1 )
        #events.Electron = ak.pad_none(events.Electron, 1, axis=1) 
        #req_ele = (ak.count(events.Electron.pt, axis=1) == 1)
        

	selection.add('onelep',(nmuons==1) & (nelectrons==1))

        ## Jet cuts
	njets = selection.add("jets", ak.sum((events.Jet.pt > 25) & (abs(events.Jet.eta) <= 2.4) & 
	((events.Jet.puId > 6 & (events.Jet.pt < 50)) | (events.Jet.pt > 50)) & events.Jet.jetId >= 2 & events.Jet.isTight ), axis=1)
       # req_jets = (ak.count(events.Jet.pt, axis=1) >= 2)    

	selection.add('njets', (njets>=2)) 

	events.Jet = events.Jet[(events.Jet.pt > 25) & (abs(events.Jet.eta) <= 2.4) & ((events.Jet.puId > 6 & (events.Jet.pt < 50)) | (events.Jet.pt > 50)) & events.Jet.jetId >= 2 & events.Jet.isTight ]
        
	
	selection.add("opposite_charge", ak.pad_none( (events.Electron,2)[:, 0].charge * (events.Muon,2)[:, 0].charge == -1), axis = 1 )
        #req_opposite_charge = events.Electron[:, 0].charge * events.Muon[:, 0].charge == -1
       	selection.add("opposite_charge_ee", ak.pad_none( (events.Electron,2)[:, 0].charge * (events.Electron,2)[:, 0].charge == -1), axis = 1 )
	selection.add("opposite_charge_mm", ak.pad_none( (events.Muon,2)[:, 0].charge * (events.Muon,2)[:, 0].charge == -1), axis = 1 )
 
        #event_level = req_trig & req_muon & req_ele & req_opposite_charge & req_jets
        
	 
        # Selected
        #selev = events[event_level]    

        #########
        
        # Per electron
        #el_eta   = (abs(selev.Electron.eta) <= 2.4)
        #el_pt    = selev.Electron.pt > 30
        #el_level = el_eta & el_pt
	
        
        # Per muon
        #mu_eta   = (abs(selev.Muon.eta) <= 2.4)
        #mu_pt    = selev.Muon.pt > 30
        #mu_level = mu_eta & mu_pt
        
        # Per jet
        #jet_eta    = (abs(selev.Jet.eta) <= 2.4)
        #jet_pt     = selev.Jet.pt > 25
        #jet_pu     = ( ((selev.Jet.puId > 6) & (selev.Jet.pt < 50)) | (selev.Jet.pt > 50) ) 
        #jet_id     = selev.Jet.jetId >= 2 
        #jet_tight     = selev.Jet.isTight # & not(selev.Jet.isTightLeptonVeto())
        #jet_level  = jet_pu & jet_eta & jet_pt & jet_id & jet_tight

        # b-tag twiki : https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation102X
        selection.add("bjet_disc_t", ak.max(events.Jet.btagDeepB, axis=1, mask_identity=False) > 0.7264) # L=0.0494, M=0.2770, T=0.7264
        selection.add("bjet_disc_m", ak.max(events.Jet.btagDeepB, axis=1, mask_identity=False) > 0.2770) # L=0.0494, M=0.2770, T=0.7264
        selection.add("bjet_disc_l", ak.max(events.Jet.btagDeepB, axis=1, mask_identity=False) > 0.0494) # L=0.0494, M=0.2770, T=0.7264
        #"bjet_level_t" = jet_level & bjet_disc_t
        #"bjet_level_m" = jet_level & bjet_disc_m
        #"bjet_level_l" = jet_level & bjet_disc_l
        
        #sel    = selev.Electron[el_level]
        #smu    = selev.Muon[mu_level]
        #sjets  = selev.Jet[jet_level]
        #sbjets_t = selev.Jet[bjet_level_t]
        #sbjets_m = selev.Jet[bjet_level_m]
        #sbjets_l = selev.Jet[bjet_level_l]

        if not isRealData: weights.add('genweight', events.genWeight)
        
	regions = {
		'default': ['trigger', 'onelep', 'oppositecharge', 'njets', 'jets']

	}
        # output['pt'].fill(dataset=dataset, pt=selev.Jet.pt.flatten())
        # Fill histograms dynamically  
        for histname, h in output.items():
            if (histname not in self.jet_hists) and (histname not in self.deepcsv_hists): continue
            # Get valid fields perhistogram to fill
            fields = {k: ak.flatten(sjets[k], axis=None) for k in h.fields if k in dir(sjets)}
            h.fill(dataset=dataset, weight=weights.weight(), **fields)


        def flatten(ar): # flatten awkward into a 1d array to hist
            return ak.flatten(ar, axis=None)

        def num(ar):
            return ak.num(ak.fill_none(ar[~ak.is_none(ar)], 0), axis=0)

	for region, cuts in regions.items():

		allcuts = set([])
		cut = selection.all(*allcuts)

	        output['njet'].fill(dataset=dataset, region=region, cut=0, weight=weights.weight()[cut], njet=normalize(ak.num(sjets),cut))
	        #output['nbjet_t'].fill(dataset=dataset, weight=weights.weight(), nbjet_t=flatten(ak.num(sbjets_t)))
	        #output['nbjet_m'].fill(dataset=dataset, weight=weights.weight(), nbjet_m=flatten(ak.num(sbjets_m)))
	        #output['nbjet_l'].fill(dataset=dataset, weight=weights.weight(), nbjet_l=flatten(ak.num(sbjets_l)))
	        output['nel'].fill(dataset=dataset, region=region,  cut=0, weight=weights.weight()[cut], nel=normalize(ak.num(sel),cut))
	        output['nmu'].fill(dataset=dataset,   region=region,  cut=0, weight=weights.weight()[cut], nmu=normalize(ak.num(smu),cut))
	
	        output['lelpt'].fill(dataset=dataset, region=region,  cut=0, weight=weights.weight()[cut], lelpt=normalize(selev.Electron[:, 0].pt,cut))
	        output['lmupt'].fill(dataset=dataset, region=region,  cut=0, weight=weights.weight()[cut], lmupt=normalize(selev.Muon[:, 0].pt,cut))
	        output['ljpt'].fill(dataset=dataset,  region=region,  cut=0, weight=weights.weight()[cut], ljpt=normalize(selev.Jet[:, 0].pt,cut))
	        output['sljpt'].fill(dataset=dataset, region=region,   cut=0, weight=weights.weight()[cut], sljpt=normalize(selev.Jet[:, 1].pt,cut))

		for i, cut in eumerate(cuts):
		allcuts.add(cut)
		cut = selection.all(*allcuts)
		
			output['njet'].fill(dataset=dataset, region=region, cut = i+1, weight=weights.weight()[cut], njet=normalize(ak.num(sjets),cut))
	        	output['nel'].fill(dataset=dataset, region=region,  cut=i+1, weight=weights.weight()[cut], nel=normalize(ak.num(sel),cut))
	        	output['nmu'].fill(dataset=dataset,   region=region,  cut=i+1, weight=weights.weight()[cut], nmu=normalize(ak.num(smu),cut))
	
	        	output['lelpt'].fill(dataset=dataset, region=region,  cut=i+1, weight=weights.weight()[cut], lelpt=normalize(selev.Electron[:, 0].pt,cut))
	        	output['lmupt'].fill(dataset=dataset, region=region,  cut=i+1, weight=weights.weight()[cut], lmupt=normalize(selev.Muon[:, 0].pt,cut))
	        	output['ljpt'].fill(dataset=dataset,  region=region,  cut=i+1, weight=weights.weight()[cut], ljpt=normalize(selev.Jet[:, 0].pt,cut))
	        	output['sljpt'].fill(dataset=dataset, region=region,   cut=i+1, weight=weights.weight()[cut], sljpt=normalize(selev.Jet[:, 1].pt,cut))

        return output

	def fill(region):
		selections = regions[region]
		cut = selection.all(*selections)
		weight = weights.weight()[cut]

#    def build_lumimask(filename):
#        from coffea.lumi_tools import LumiMask
#	with importlib.resources.path("BTVNanoCommissioning.data", filename) as path:
#	return LumiMask(path)

#    lumiMasks = {
#	'2016': build_lumimask('Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt'), 
#        }

    def postprocess(self, accumulator):
        return accumulator
