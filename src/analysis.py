import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import scipy
import scipy.stats

import cobra


import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

from csv import writer
import emll

#import pymc as pm
# import os

########################################

def plot_ELBO_convergence(pickleJar, runName, iter):
    approx = pickleJar['approx']
    
    with sns.plotting_context('notebook', font_scale=1.2):

        fig = plt.figure(figsize=(5,4),dpi=100)
        plt.plot(approx.hist + 30, '.', rasterized=True, ms=1)
        plt.yscale("log")
        # plt.ylim([1E2, 1E7])
        plt.xlim([0, iter])
        sns.despine(trim=True, offset=10)

        plt.ylabel('-ELBO')
        plt.xlabel('Iteration')
        plt.title(f'{runName} ELBO convergence')
        # plt.tight_layout()
        plt.savefig(f'{runName}_elbo.svg', transparent=True, dpi=200)

def calculate_elasticity_medians(pickleJar, runName):
    trace = pickleJar['trace']
    e_labels = pickleJar['e_labels']
    
    postEx = np.squeeze(trace['posterior']['Ex'].to_numpy()) # (1000, 91, 80)
    postEy = np.squeeze(trace['posterior']['Ey'].to_numpy()) # (1000, 91, 14)

    a = postEx.reshape((-1, postEx.shape[1]*postEx.shape[2]))
    b = postEy.reshape((-1, postEy.shape[1]*postEy.shape[2]))
    medians = pd.DataFrame(np.hstack([a, b]), columns=e_labels).median().to_numpy()
    with open(f'{runName}_elasticity_medians.csv', 'a', newline='') as f:
        writer(f).writerow(['median'] + list(medians))
        f.close()

def calculate_FCC_medians(pickleJar, runName):
    trace = pickleJar['trace']
    e_labels = pickleJar['e_labels']
    
    postEx = np.squeeze(trace['posterior']['Ex'].to_numpy()) # (1000, 91, 80)
    postEy = np.squeeze(trace['posterior']['Ey'].to_numpy()) # (1000, 91, 14)

    a = postEx.reshape((-1, postEx.shape[1]*postEx.shape[2]))
    b = postEy.reshape((-1, postEy.shape[1]*postEy.shape[2]))
    medians = pd.DataFrame(np.hstack([a, b]), columns=e_labels).median().to_numpy()
    with open(f'{runName}_elasticity_medians.csv', 'a', newline='') as f:
        writer(f).writerow(['median'] + list(medians))
        f.close()

    medians = pd.DataFrame(np.squeeze(medians), index=cc_index, columns=r_labels)
    medians.to_csv(runName + 'FCC_medians.csv')
    

def save_sampled_elasticities(pickleJar, runName):
    trace = pickleJar['trace']
    ex_labels = pickleJar['ex_labels']
    ey_labels = pickleJar['ey_labels']

    print(ex_labels.shape)
    print(ey_labels.shape)

    e_labels = np.hstack((ex_labels, ey_labels))
    
    e_labels = np.hstack((ex_labels, ey_labels))
    
    Ex_hdi = az.hdi(trace['posterior']['Ex'])['Ex'].to_numpy() #(91, 80, 2)
    Ey_hdi = az.hdi(trace['posterior']['Ey'])['Ey'].to_numpy() #(91, 14, 2)

    print(Ex_hdi.shape)
    print(Ey_hdi.shape)


    print(Ex_hdi.shape)
    print(Ey_hdi.shape)

    ex = Ex_hdi.reshape((Ex_hdi.shape[0]*Ex_hdi.shape[1],-1))
    ey = Ey_hdi.reshape((Ey_hdi.shape[0]*Ey_hdi.shape[1],-1))
    e_all = np.transpose(np.vstack([ex, ey]))
    e_df_vi = pd.DataFrame(e_all, columns=e_labels)
    e_df_vi.to_csv(f'{runName}_predicted_elasticities.csv')

def calculate_FCC_hdi(pickle_jar, runName, medians, cc_type='fcc'):
    """
    Ex_hdi is the hdi of the posterior Ex trace as a numpy array
    """
    trace = pickle_jar['trace']
    # trace_prior = pickle_jar['trace_prior']?????
    ll = pickle_jar['ll']
    m_labels = pickle_jar['m_labels']
    r_labels = pickle_jar['r_labels']
    y_labels = pickle_jar['r_labels']

    medians = medians.reshape((1, len(r_labels),-1))

    """
    if cc_type=='mcc':
        cc_mb = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in a])   
        cc_prior = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in b]) 
        medians = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in medians])
        hdi_upper = pd.DataFrame(cc_mb[0], index=m_labels, columns=r_labels).rename_axis(('Reactions'))
        hdi_lower = pd.DataFrame(cc_mb[1], index=m_labels, columns=r_labels).rename_axis(('Reactions'))
        cc_hdi = pd.concat({'upper': hdi_upper, 'lower': hdi_lower}, names=['hdi']).swaplevel()
        cc_hdi =pd.pivot_table(cc_hdi, index=['Reactions', 'hdi'])
        cc_index = m_labels+y_labels
    """

    # elif cc_type=='fcc':
    cc = np.array([ll.flux_control_coefficient(Ex=ex) for ex in trace['posterior']['Ex']['Ex'].to_numpy()])
    cc_prior = np.array([ll.flux_control_coefficient(Ex=ex) for ex in trace['prior']['Ex']['Ex'].to_numpy()])
    medians = np.array([ll.flux_control_coefficient(Ex=ex) for ex in medians]) 
    hdi_upper = pd.DataFrame(cc[0], index=r_labels, columns=r_labels).rename_axis(('Reactions'))
    hdi_lower = pd.DataFrame(cc[1], index=r_labels, columns=r_labels).rename_axis(('Reactions'))
    cc_hdi = pd.concat({'upper': hdi_upper, 'lower': hdi_lower}, names=['hdi']).swaplevel()
    cc_hdi =pd.pivot_table(cc_hdi, index=['Reactions', 'hdi'])
    cc_index = r_labels

    cc_hdi.to_csv(runName + f'_{cc_type.upper()}s_hdi.csv')

    #return cc_df


def plot_FCC_distrb(cc_df, cc_type, results_dir, dataset_name):
    
    fig = plt.figure(figsize=(16, 8))

    my_pal = {"Prior": ".8", "ADVI":"b"}

    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frameon=False, sharex=ax, sharey=ax)

    sns.violinplot(
        x='Reactions', y=cc_type, hue='Type', data=cc_df[cc_df.Type == 'Prior'],
        scale='width', width=0.5, legend=False, zorder=0,
        color='1.', ax=ax, saturation=1., alpha=0.01)

    plt.setp(ax.lines, color='.8')
    plt.setp(ax.collections, alpha=.5, label="")

    sns.violinplot(
        x='Reactions', y=cc_type, hue='Type', data=cc_df,
        scale='width', width=0.8, hue_order=['ADVI'],
        legend=False, palette=my_pal, zorder=3, ax=ax2)

    phandles, plabels = ax.get_legend_handles_labels()
    handles, labels = ax2.get_legend_handles_labels()
    ax.set(xlabel=None)
    ax.legend().remove()
    ax2.legend().remove()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax2.tick_params(axis='x', labelsize=8, rotation=90)
    
    ax2.legend(phandles + handles, plabels + labels, loc='upper center', ncol=4, fontsize=13)
    ax.set_ylim([-1.5, 1.5])

    ax.axhline(0, ls='--', color='.7', zorder=0)
    sns.despine(trim=True)

    plt.suptitle(dataset_name + f' Predicted {cc_type.upper()}s: 4ACA', y=1)

    fig.savefig(results_dir + f'{dataset_name}-plotted_{cc_type}s.svg', transparent=True)


def print_predicted_Ex_values(pickleJar, output_path):
    t = np.squeeze(pickleJar['trace']['posterior']['Ex'].to_numpy())
    dim=t.shape
    traceDF = pd.DataFrame(t.reshape(dim[0]*dim[1], dim[2])) 
    traceDF.to_csv(f'{output_path}_PredictedExs.csv')

