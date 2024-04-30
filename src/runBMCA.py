import sys
import os

os.environ['MKL_NUM_THREADS'] = '20'
os.environ['OMP_NUM_THREADS'] = '20'

OUTPUT_FOLDER = 'output/'
  
# import pickle5 as pickle
import cloudpickle as pickle

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import arviz as az

import pymc as pm
import aesara
import aesara.tensor as at
aesara.config.exception_verbosity='high'

import cobra
import emll


## helper functions
def create_gprdict(model):   
    gpr_dict = dict()
    for rxn in model.reactions:
        if rxn.gene_reaction_rule:
            temp = set()
            for x in [x.strip('() ') for x in rxn.gene_reaction_rule.split(' or ')]:
                temp.add(frozenset(y.strip('() ') for y in x.split(' and ')))
            gpr_dict[rxn.id] = temp
    return gpr_dict

def transcript_value_for_rxn(model, transcriptomics_df, rxn):
    final_transcript_value = 0
    gene_ids = []
    for parallel_gene in create_gprdict(model)[rxn.id]:
        transcript_values = []
        for gene in parallel_gene:
            if gene in transcriptomics_df.index:
                transcript_values.append(transcriptomics_df.loc[gene])
                # transcript_values.append(transcriptomics_df.loc[gene].to_numpy()[0])
#                 print(transcriptomics_df.loc[gene].to_numpy()[0])
            else:
                transcript_values.append(np.inf)
            min_transcript_val = np.min(transcript_values)
        final_transcript_value = final_transcript_value + min_transcript_val
#         if final_transcript_value==newinfbound:
#             display(rxn.id)
#             gene_ids.append(rxn.id)
    return final_transcript_value

def runBMCA(runName, N_ITERATIONS=50000):

    # import boundary-implicit, cobra-friendly version of model
    cobra_model = cobra.io.read_sbml_model('src/models/iJN1463_JS.xml')

    v_df = pd.read_csv("src/data/iJN1463_JS_eflux2_flux.csv", index_col='Unnamed: 0').transpose()

    # remove reactions from dataset that are not in the cobra_model
    v_df = v_df[[i for i in v_df.columns if i in [ii.id for ii in cobra_model.reactions]]]

    # import data
    transcriptomics_df = pd.read_csv("src/data/putida_RNAseq_data.csv", index_col='strains')[['genes', 'Value']]
    transcriptomics_df = pd.pivot_table(transcriptomics_df, values='Value', index='genes', columns='strains').round(2)

    # check if reaction has a gene reaction rule
    geneRxns = [i.id for i in cobra_model.reactions if i.gene_reaction_rule]

    # get the transcriptomics values for each reaction listed in geneRxns
    ds = []
    for strain in transcriptomics_df.columns:
        # get xscript values for each reaction
        transcriptValues = [transcript_value_for_rxn(cobra_model, transcriptomics_df[strain], cobra_model.reactions.get_by_id(i)) for i in geneRxns]
        transcriptValues = dict(zip(geneRxns, transcriptValues))
        # get rid of reactions with infinite transcription
        transcriptValues = {k: v for k, v in transcriptValues.items() if v != np.inf}
        ds.append(transcriptValues)
    e_df = pd.DataFrame(ds, index=transcriptomics_df.columns)

    # perform additive smoothing
    ADD = 1.01
    e_df = np.log(e_df.astype('float') + ADD)

    # importing external metabolite concentration data
    external_met_file = 'src/data/putida_ext_metabolomics_data.csv'
    y_df = pd.read_csv(external_met_file)#.astype(float)
    y_df = y_df.set_index('Line Description')
    y_df.rename(columns={'4-Aminocinnamic acid': '4aca_e', 
                            '4-Hydroxycinnamic acid': 'pHCA_e',
                            '4-aminobenzoic acid': '4abz_e',
                            'CINNAMIC ACID': 'cinm_e',
                            'p-Aminophenylalanine': '4aPhe_e'}, inplace=True)

    # zero out any negative concentration values
    y_df[y_df < 0] = 0

    # Get rid of the strains in the dataset that are not present in the 
    # transcriptomics dataset.
    y_df = y_df.loc[e_df.index]

    # perform additive smoothing
    y_df = np.log(y_df.astype('float') + ADD)

    # Designate the reference strain
    
    #Since we are maximizing for 4aca_e, our reference strain will be the 
    #strain that produces the most 4aca_e, which is `'pACA production 3 
    #scRNA positive control'`
    
    ref_strain = 'pACA production 3 scRNA positive control'
    v_star = v_df.loc[ref_strain].values
    e_star = e_df.loc[ref_strain].values
    y_star = y_df.loc[ref_strain].values

    # Checking for 0 or negative values in inputs
    assert len(v_star[v_star == 0])==0 # fluxes can be (-) but concentrations cannot
    assert len(e_star[e_star <= 0])==0
    assert len(y_star[y_star <= 0])==0

    assert (len(e_df.values[e_df.values <= 0]) == 0)
    assert (len(y_df.values[y_df.values <= 0]) == 0)

    # Normalizing the data to the reference strain
    yn = y_df.divide(y_star)
    vn = v_df.divide(v_star)
    en = e_df.divide(e_star)

    assert (vn == 0).sum().sum()==0
    assert (en <= 0).sum().sum()==0
    assert (yn <= 0).sum().sum()==0

    N = cobra.util.create_stoichiometric_matrix(cobra_model)

    # Correct negative flux values at the reference state
    N[:, v_star < 0] = -1 * N[:, v_star < 0]
    v_star = np.abs(v_star)

    assert np.isclose(np.all(np.matmul(N, v_star)), 0), "data does not describe steady state"
    assert(len(np.where(N@v_star >1e-6)[0]) == 0)

    # Load the Cobra version of the model
    model = cobra.io.read_sbml_model('src/models/iJN1463_JS.xml') 
    model.tolerance = 1e-9

    # Set up the Bayesian inference
    # reactions and metabolite compartments
    r_compartments = [r.compartments if 'e' not in r.compartments else 't' for r in model.reactions]

    for rxn in model.exchanges:
        r_compartments[model.reactions.index(rxn)] = 't'

    m_compartments = [m.compartment for m in model.metabolites]
    internal_mets = [i for i in model.metabolites if i.compartment!='e']
    external_mets = [i for i in model.metabolites if i.compartment=='e']
    rxnNames = [i.id for i in model.reactions]
    v_inds = np.arange(0,len(v_star))
    n_exp = v_df.shape[0]

    # Establish labels for metabolite and reaction names
    m_labels = [m.id for m in model.metabolites]
    r_labels = [r.id for r in model.reactions]
    # x_labels = [i.id for i in internal_mets]
    y_labels = [i.id for i in external_mets]

    ex_labels = np.array([['$\epsilon_{' + '{0},{1}'.format(rlabel, mlabel) + '}$'
                        for mlabel in m_labels if mlabel not in y_labels] for rlabel in r_labels]).flatten()
    ey_labels = np.array([['$\epsilon_{' + '{0},{1}'.format(rlabel, mlabel) + '}$'
                        for mlabel in y_labels] for rlabel in r_labels]).flatten()

    # e_labels = np.hstack((ex_labels, ey_labels))

    # Set up elasticity matrices
    Ex = emll.create_elasticity_matrix(model)
    Ey = np.zeros((len(model.reactions), len(external_mets)))

    ey_indices = {}
    for met in y_labels:
        ey_indices[met]=[model.reactions.index(rxn) for rxn in model.metabolites.get_by_id(met).reactions]
    for i, met in enumerate(ey_indices.keys()):
        for ii in ey_indices[met]:
            Ey[ii, i] = 1

    # make copy of order of metabolites in matrices
    with open(f'{OUTPUT_FOLDER}order.list', 'w') as f:
        f.write('// REACTIONS\n')
        for i, ii in enumerate(r_labels):
            f.write(str(i) + ' ' + ii + '\n')
        f.write('\n// INTERNAL METABOLITES\n')
        for i, ii in enumerate(m_labels):
            f.write(str(i) + ' ' + ii + '\n')
        f.write('\n// EXTERNAL METABOLITES\n')
        for i, ii in enumerate(y_labels):
            f.write(str(i) + ' ' + ii + '\n')

    # Setting up the PyMC model
    ll = emll.LinLogLeastNorm(N,Ex,Ey,v_star, driver = 'gelsy')
    from emll.util import initialize_elasticity

    with pm.Model() as pymc_model:
        # Initialize elasticities
        Ex_t = pm.Deterministic('Ex', initialize_elasticity(N, 'ex', b=0.05, sd=1, alpha=5, m_compartments=m_compartments,
            r_compartments=r_compartments))
        Ey_t = pm.Deterministic('Ey', initialize_elasticity(-Ey.T, 'ey', b=0.05, sd=1, alpha=5))

        known_e_inds = []
        unknown_e_inds = []
        for i, e in enumerate(rxnNames):
            if e in e_df.columns:
                known_e_inds.append(i)
            else: 
                unknown_e_inds.append(i)
        e_inds = np.hstack([known_e_inds, unknown_e_inds]).argsort()

    with pymc_model:
        #Protein Expression Priors
        e_measured = pm.Normal('e_measured', mu=en, sigma=0.1, shape=(n_exp, len(known_e_inds))) # (41, 524)
        e_unmeasured = pm.Laplace('e_unmeasured', mu=0, b=0.1, shape = (n_exp, len(unknown_e_inds))) # 41, 43

        en_t = aesara.tensor.concatenate([e_measured, e_unmeasured], axis=1)[:, e_inds]
        pm.Deterministic('en_t', en_t)

    with pymc_model:
        xn_t = pm.Normal('xn_t', mu=0, sigma=10, shape=(Ex.shape[1], n_exp),
                        initval=0.1 * np.random.randn(Ex.shape[1], n_exp))

    known_y_inds = []
    omitted_y_inds = []
    for i, y in enumerate(external_mets):
        if y.id in y_df.columns:
            known_y_inds.append(i)
        else: 
            omitted_y_inds.append(i)
    y_inds = np.hstack([known_y_inds, omitted_y_inds]).argsort()
                    
    # y variable should be split because there are some observations
    with pymc_model:
        y_unmeasured = pm.Normal('y_unmeasured', mu=3, sigma=0.1, shape = (len(omitted_y_inds), n_exp))

    with pymc_model:
        # External Metabolite and Flux Steady State priors
        y_ss, vn_ss = ll.steady_state_aesara(Ey_t, Ex_t, en_t, xn_t.T, n_exp=n_exp)
        pm.Deterministic('y_ss', y_ss)
        pm.Deterministic('vn_ss', vn_ss)

    with pymc_model:
        #External Metabolite Priors
        y_clip = y_ss[:,list(range(len(y_df.columns)))].clip(-1.5, 1.5)
        y_obs = pm.Normal('y_measured', mu=y_clip, sigma=0.1, shape=(n_exp, yn.shape[1]), observed=yn.clip(lower=-1.5, upper=1.5)) # sigma could be y_err
        # y_measured = pm.Normal('y_measured', mu=3, sigma=0.1, observed=yn)# shape=(n_exp, len(y_inds))
        vn_t = pm.Normal('vn_t', mu=vn_ss, sigma=0.1, observed=vn.clip(lower=-1.5, upper=1.5))

    # prior predictive check
    with pymc_model:
        RANDOM_SEED = np.random.seed(1)
        trace_prior = pm.sample_prior_predictive(samples=1000, random_seed=RANDOM_SEED)

    # trace_prior['prior']['ex_kinetic_entries']
    priorEx = np.squeeze(trace_prior['prior']['Ex'].to_numpy()) # (500, 13, 8)

    met_priors = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in priorEx]) 
    met_priors_hdi = az.hdi(az.convert_to_dataset(met_priors[np.newaxis,:])) #(13, 13, 2)

    FCC_priors = np.array([ll.flux_control_coefficient(Ex=ex) for ex in priorEx]) # (500, 13, 13)
    FCC_priors_hdi = az.hdi(az.convert_to_dataset(FCC_priors[np.newaxis,:])) #(13, 13, 2)

    # Running the ADVI
    
    with pymc_model:
        advi = pm.ADVI()
        tracker = pm.callbacks.Tracker(
            mean = advi.approx.mean.eval,
            std = advi.approx.std.eval
        )
        approx = advi.fit(
            n=N_ITERATIONS, 
            callbacks = [tracker],
            obj_optimizer=pm.adagrad_window(learning_rate=5E-3), 
            total_grad_norm_constraint=0.7,
            obj_n_mc=1)

    SAMPLE_DRAWS = 1000
    
    print("sampling draws")
    with pymc_model:
        trace_vi = approx.sample(draws=SAMPLE_DRAWS, random_seed=1) 
    
    print("sampling posterior")
    with pymc_model:
        ppc_vi = pm.sample_posterior_predictive(trace_vi, random_seed=1)
    
    # ppc_vi['posterior_predictive']['v_hat_obs']

    ## PICKLE PICKLE PICKLE
    print('pickling trace')
    
    #with open(f'/putidabmca/output/{runName}.pgz','wb') as f:
    pickle.dump({'advi': advi,
        'approx': approx,
        'trace': trace_vi,
        'trace_prior': trace_prior,
        'ex_labels': ex_labels,
        'ey_labels': ey_labels,
        'r_labels': r_labels,
        'm_labels': m_labels,
        'y_labels': y_labels,
        'll': ll}, file=open(f'{OUTPUT_FOLDER}{runName}.pgz', "wb"))
