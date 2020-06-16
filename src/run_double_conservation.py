import sys
import doublecons as dc
import argparse
import os
import logging
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.cm as cm
import matplotlib

def read_input_file(args):
    """
    this function is to parse and read the input file.
    everything will be stored in parser arguments

    parse param added:
    args.ligands    -   dictionary of ligand class
    args.msa        -   multiple sequence alignment inside MSA class
    args.msta       -   multiple structure alignment inside MSA class
    args.receptor   -   receptor class
    :param args:
    :return:
    """
    # PCA has different input type
    if args.test_type == "PCA":
        return
    # read standard input file
    ligands = {}
    ref = args.concon_reference

    with open(args.input_file) as infile:
        for line in infile:
            if line[0] != "#":
                line = line.strip().split()
                type = line[0]

                # new ligand
                if type == "LIGAND":
                    p, name, pdb = line[1:]
                    this_lig = dc.Ligand(name, pdb=pdb)
                    this_lig.prot_msa = dc.create_msa(open(p))
                    assert name not in ligands, "I am not overwriting ligand {}".format(name)
                    ligands[name] = this_lig
                # new MSA
                if type == "MSA":
                    p = line[1]
                    msa = dc.MSA("MSA", reference=ref)
                    msa.msa = dc.create_msa(open(p))
                    args.msa = msa
                # new MSTA
                if type == "MSTA":
                    p = line[1]
                    msta = dc.MSA("MSTA", reference=ref)
                    msta.msa = dc.create_msa(open(p))
                    args.msta = msta
                # new receptor
                if type == "RECEPTOR":
                    name, p = line[1:]
                    receptor = dc.Receptor(name, pdb=pdb, reference="Hsap")
                    receptor.prot_msa = dc.create_msa(open(p))
                    args.receptor = receptor

        # add ligands
        args.ligands = ligands

def con_con_comparison(args):
    """
    cross conservation comparison

    :param args:
    :return:
    """
    rp = args.concon_reference          # reference ligand name
    prot = args.ligands[rp]             # reference ligand
    seq = prot.get_ref_seq()            # reference ligand sequence
    msta = args.msta                    # multiple structure alignment
    msa = args.msa                      # multiple sequence alignment
    contest = args.conservation_test    # conservation test type
    algtest = args.alignment_test       # conservation for ligand alignment type

    # data preparation
    cs = pd.DataFrame()
    # add separating variable when it is 7 lumps of data
    sepvar = 0
    if algtest == "id":
        sepvar = 0.04
    cs["MSA"] = [round(x, 2) + sepvar for x in msa.get_cons_scores(algtest)]
    cs["MSTA"] = [round(x, 2) - sepvar for x in msta.get_cons_scores(algtest)]
    cs[prot.name] = prot.get_cons_scores(contest)
    cs.insert(0, 'POS', [str(x) + seq[x - 1] for x in range(1, 1 + len(cs))])

    # plotting
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)  # A4 size

    def plotter(fig, ax, alg_label, marker, pname):

        # colormap
        cmap = matplotlib.cm.get_cmap('coolwarm')
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(cs[alg_label]))
        colors = [cmap(normalize(value)) for value in range(len(cs[alg_label]))]

        #scatter plot
        sb.regplot(alg_label, pname, cs, scatter=True, fit_reg=False, ax=ax, label=alg_label, marker=marker, scatter_kws={"color":colors})

    plotter(fig, ax, "MSA", 'd', prot.name)
    plotter(fig, ax, "MSTA", 's', prot.name)

    # labels
    plt.ylabel("Evolutionary Conservation")
    plt.xlabel("Ligands Alignments Conservation")
    plt.legend(loc=4)

    # # colorbar
    cmap = matplotlib.cm.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(cs["MSA"]))
    cax, _ = matplotlib.colorbar.make_axes(ax, orientation="horizontal", aspect=50)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, orientation="horizontal")
    plt.xlabel("AA position")

    # plot diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
    ax.plot(lims, lims, ls="--", c=".3")
    # in case it is 7 lumps of X axis
    if algtest == "id":  # it is 7 blocks
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # float representation to 2 decimals
        ax.bar(np.arange(0, 1.01, 1 / 6), [ax.get_ylim()[1]] * 7, width=[0.15] * 7, color="grey", alpha=0.3)  # backgroun bars
        plt.xticks(np.arange(0, 1.01, 1 / 6))  # ligands xticks

    # saving
    plt.savefig("{}/{}concon.png".format(args.output_path, prot.name), format='png', dpi=800)
    # save text output also
    with open("{}/{}concon.csv".format(args.output_path, prot.name), "w") as outfile:
        cs.to_csv(outfile, sep="\t")

def dir_prediction(args):
    """
    Divergence Inducing Residue Prediction
    Basically like cross-conservation but averaging over all paralogs and correcting for coevolution

    :param args:
    :return:
    """
    rp = args.concon_reference          # reference ligand name
    prot = args.ligands[rp]             # reference ligand
    seq = prot.get_ref_seq()            # reference ligand sequence
    msta = args.msta                    # multiple structure alignment
    msa = args.msa                      # multiple sequence alignment
    contest = args.conservation_test    # conservation test type
    algtest = args.alignment_test       # conservation for ligand alignment type

    # data preparation
    cs = pd.DataFrame()
    # add separating variable when it is 7 lumps of data
    sepvar = 0
    if algtest == "id":
        sepvar = 0.04
    cs["MSA"] = [round(x, 2) + sepvar for x in msa.get_cons_scores(algtest)]
    cs["MSTA"] = [round(x, 2) - sepvar for x in msta.get_cons_scores(algtest)]
    relative_conservation = ligand_table(args)
    # cs[prot.name] = prot.get_cons_scores(contest)
    # now I need one evo cons from MSA alignment average and one from MSTA alignment average
    cs["EVO cons MSA"] = average_cons(relative_conservation,"MSA")
    cs["EVO cons MSTA"] = average_cons(relative_conservation,"MSTA")
    cs.insert(0, 'POS', [str(x) + seq[x - 1] for x in range(1, 1 + len(cs))])

    # plotting
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)  # A4 size

    def plotter(fig, ax, alg_label, marker, pname):

        # colormap
        cmap = matplotlib.cm.get_cmap('coolwarm')
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(cs[alg_label]))
        colors = [cmap(normalize(value)) for value in range(len(cs[alg_label]))]

        #scatter plot
        sb.regplot(alg_label, pname, cs, scatter=True, fit_reg=False, ax=ax, label=alg_label, marker=marker, scatter_kws={"color":colors, "alpha": 0.5, "linewidth":0.5, "edgecolor":"black"})
        # labels
        dpos = set()

        def label_point(x, y, val, ax, dpos, setpos=set()):
            a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
            for i, point in a.iterrows():
                # point['x'] += 0.01
                # i = 0
                # while (point['x'], point['y']) in dpos:
                #     if i == 1:
                #         point['x'] -= 0.04
                #         point['y'] -= 0.01
                #         i = -1
                #     if i == 1 and round(point['x'], 2) == 1:
                #         point['x'] -= 0.04
                #         point['y'] -= 0.01
                #         i = -1
                #     i += 1
                #     point['x'] += 0.02
                if not setpos or any([str(pos) in point['val'] for pos in setpos]):
                    dpos.add((point['x'], point['y']))
                    ax.text(point['x'], point['y'], str(point['val']), fontsize=12)

        setpos = {32,46,48,50}
        label_point(cs[alg_label], cs["EVO cons {}".format(alg_label)], cs.POS, plt.gca(), dpos, setpos)
        label_point(cs[alg_label], cs["EVO cons {}".format(alg_label)], cs.POS, plt.gca(), dpos, setpos)

    plotter(fig, ax, "MSA", 'd', "EVO cons MSA")
    plotter(fig, ax, "MSTA", 's', "EVO cons MSTA")

    # labels
    plt.ylabel("Average Evolutionary Conservation")
    plt.xlabel("Ligands Alignments Conservation")
    plt.legend(loc=4)

    # # colorbar
    cmap = matplotlib.cm.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(cs["MSA"]))
    cax, _ = matplotlib.colorbar.make_axes(ax, orientation="horizontal", aspect=50)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, orientation="horizontal")
    plt.xlabel("AA position")

    # plot diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
    ax.plot(lims, lims, ls="--", c=".3")
    # in case it is 7 lumps of X axis
    if algtest == "id":  # it is 7 blocks
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # float representation to 2 decimals
        ax.bar(np.arange(0, 1.01, 1 / 6), [ax.get_ylim()[1]] * 7, width=[0.15] * 7, color="grey", alpha=0.3)  # backgroun bars
        plt.xticks(np.arange(0, 1.01, 1 / 6))  # ligands xticks

    # remove top and right
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # saving
    plt.savefig("{}/{}dirp.png".format(args.output_path, prot.name), format='png', dpi=800)

    # save text output also but add average
    # average MSA/MSTA then distance with
    cs["avg DIR score"] = calculate_dirp_score(cs,["MSA","EVO cons MSA"],["MSTA","EVO cons MSTA"])
    with open("{}/{}dirp.csv".format(args.output_path, prot.name), "w") as outfile:
        cs.to_csv(outfile, sep="\t")

def calculate_dirp_score(df, msa_id=["MSA","EVO cons MSA"], msta_id=["MSTA","EVO cons MSTA"]):
    """average msa and msta score (distance from diagonal)"""

    diag = lambda x,y: np.abs(x-y) / np.sqrt(2)
    MSA_score = diag(df[msa_id[0]],df[msa_id[1]])
    MSTA_score = diag(df[msta_id[0]],df[msta_id[1]])
    return np.mean([MSA_score,MSTA_score], axis=0)


def average_cons(d,alg_type):
    """returns the average conservation from the dictionary of relative scores"""
    l = []
    for k in d:
        if alg_type in k:
            a = [np.nan if x == "-" else float(x) for x in d[k]]
            l.append(a)
    mean_out = np.nanmean(l,axis=0)
    return mean_out

def ligand_table(args):
    """
    :param args:
    :return:
    """
    rp = args.concon_reference          # reference ligand name
    prot = args.ligands[rp]             # reference ligand
    seq = prot.get_ref_seq()            # reference ligand sequence
    msta = args.msta                    # multiple structure alignment
    msa = args.msa                      # multiple sequence alignment
    contest = args.conservation_test    # conservation test type
    algtest = args.alignment_test       # ligand alignment test type

    # data preparation
    cs = pd.DataFrame()
    rel_cons = {}  # relative conservation is stored here

    # add referenced positions
    for m in [msa, msta]:
        for n in m.get_names():
            logging.warning("processing {}".format(n))
            assert n in args.ligands, "{} not found in ligands list: {}".format(n,args.ligands.keys())
            this_ligand = args.ligands[n]
            this_ligand_score = this_ligand.get_cons_scores(contest)
            this_name = "{}_{}".format(m.name, n)
            cs[this_name] = m.get_referenced_positions(n)
            rel_cons[this_name] = m.get_referenced_scores(n, this_ligand, this_ligand_score)

    # add also cons score
    cs["MSA"] = [round(x, 2) + 0.04 for x in msa.get_cons_scores(algtest)]
    cs["MSTA"] = [round(x, 2) - 0.04 for x in msta.get_cons_scores(algtest)]
    cs[prot.name] = prot.get_cons_scores(contest)
    for n in rel_cons:
        cs[n+"_cons_{}".format(contest)] = rel_cons[n]
    # add egf pos
    cs.insert(0, 'POS', [str(x) + seq[x - 1] for x in range(1, 1 + len(cs))])

    with open("{}/ligand_table.csv".format(args.output_path), "w") as outfile:
        cs.to_csv(outfile, sep="\t")

    return rel_cons

def coevol_test(args):

    # preparation
    rp = args.concon_reference          # reference ligand name
    prot = args.ligands[rp]             # reference ligand
    receptor = args.receptor            # receptor MSA
    msta = args.msta  # multiple structure alignment
    msa = args.msa  # multiple sequence alignment

    # receptor intra coevol
    dc.intra_coevol(receptor, "{}/{}".format(args.output_path, receptor.name), args.coevolution_test)
    # MSA - MSTA coevol
    dc.intra_coevol(msa, "{}/{}".format(args.output_path, msa.name), args.coevolution_test)
    dc.intra_coevol(msta, "{}/{}".format(args.output_path, msta.name), args.coevolution_test)
    # every ligand coevol
    for lig_prot in args.ligands.values():
        dc.intra_coevol(lig_prot, "{}/{}".format(args.output_path, lig_prot.name), args.coevolution_test)
        dc.inter_coevol(lig_prot, receptor, "{}/{}u{}intercoevo".format(args.output_path, lig_prot.name, receptor.name), args.coevolution_test)


def pca_test(args):
    """
    assume "POS" column is reference pos and residue string
    assume "MSA" and "MSTA" are columns with respective conservation scores
    in parallel, plots for MSA and MSTA
    :param args:
    :return:
    """
    assert os.path.exists(args.input_file), "You need to specify a ligand table before, consider running LTABLE test."
    tot_df = pd.read_csv(open(args.input_file),sep="\t")

    ## PCA scripts
    def run_pca(df, ref, alg_type, out_path):

        logging.warning("running PCA for {}, brace yourself".format(alg_type))
        df_lab = df["POS"].values
        df_cons = df[alg_type].values

        df = df.filter(regex="({}_.+_cons)".format(alg_type, ref))
        # print(df)
        X = []
        y = []
        # score distribution
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(8, 6))
            for lab in df.columns:
                if "_cons_" in lab:
                    # print(lab)
                    this_data = ",".join(list(df[lab].astype(str))).replace("-,", "0.0,")
                    if this_data[-1] == "-":
                        this_data = this_data[:-1] + "0.0"
                    this_data = [float(x) for x in this_data.split(",")]
                    X.append(this_data)
                    y.append(lab)
                    plt.hist(this_data, label=lab, bins=10, alpha=0.3, )
            plt.legend()
        #    plt.show()
            plt.title("{} Features score distribution".format(alg_type))
            plt.tight_layout()
            plt.savefig("{}{}_score_distro.svg".format(out_path, alg_type),format="svg", dpi=1200)
        plt.clf()

        # something else
        # zero to 1
        X_std = StandardScaler().fit_transform(np.array(X).T)
        # transformed histogram
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(8, 6))
            for i in range(len(y)):
                plt.hist(X_std.T[i], label=y[i], bins=10, alpha=0.3, )

        plt.legend()
        # plt.show()
        plt.title("{} Features normalized score distribution".format(alg_type))
        plt.tight_layout()
        plt.savefig("{}{}_norm_score_distro.svg".format(out_path, alg_type), format="svg", dpi=1200)
        plt.clf()

        sklearn_pca = sklearnPCA(n_components=2)
        Y_sklearn = sklearn_pca.fit_transform(X_std)
        # print(len(Y_sklearn))
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(6, 4))
            plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=df_cons, cmap="viridis")
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')

            # add points
            def label_point(x, y, val, ax):
                for i in range(len(x)):
                    ax.text(x[i], y[i], val[i], fontsize=5, color="red")

            label_point(Y_sklearn[:, 0], Y_sklearn[:, 1], df_lab, plt.gca())

            # plt.legend(loc='lower center')
            # plt.show()
            plt.title("{} Principal Component Analysis".format(alg_type))
            plt.tight_layout()
            plt.savefig("{}{}_PCA.svg".format(out_path, alg_type), format="svg", dpi=1200)
        plt.clf()
        # covariance
        # mean_vec = np.mean(X_std)
        # cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
        # print(cov_mat)
        cov_mat = np.cov(X_std.T)
        logging.warning('NumPy covariance matrix: \n%s' % np.cov(X_std))

        # eigenvalues
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        logging.warning('Eigenvectors \n%s' % eig_vecs)
        logging.warning('\nEigenvalues \n%s' % eig_vals)

        # testing dimension
        for ev in eig_vecs:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
        logging.warning('Everything ok!')

        # choosing eigvect
        # list of eig val, eig vect tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        # sort
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # visually confirm
        logging.warning('Eigenvalues in descending order:')
        for i in eig_pairs:
            logging.warning(i[0])

        # explained variance
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(6, 4))
            plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center', label='individual explained variance')
            plt.step(range(len(var_exp)), cum_var_exp, where='mid', label='cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.legend(loc='best')
            # plt.show()
            plt.title("{} Eigenvectors Explained Variance".format(alg_type))
            plt.tight_layout()
            plt.savefig("{}{}_explain_var.svg".format(out_path, alg_type), format="svg", dpi=1200)
        plt.clf()

    # do it for both MSA and MSTA
    run_pca(tot_df, args.concon_reference, "MSA", args.output_path)
    run_pca(tot_df, args.concon_reference, "MSTA", args.output_path)


def main(argv):

    # PARSER
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description="parser for handling input files of cons-cons pipeline")

    parser.add_argument("--input-file", required=True, help="input file contains all ligands with their MSA location"
                                                        "the MStA path and its format is very important, check example")
    parser.add_argument("--output-path", required=True, help="path to output folder, will be created if not ex")
    parser.add_argument("--test-type", required=True, help="""The type of test:
     CONCON - cross conservation graph,
     DIRP - average multiple concon, or Divergence Inducing Residue Prediction,
     LTABLE - ligand positions to reference table
     COEVOL - coevolution test between ligands and receptor
     PCA    - PCA done from the conservation scores of ligand table. needs ligtable as input""")
    parser.add_argument("--concon-reference", default="", help="the reference protein for CONCON test.")
    parser.add_argument("--conservation-test", default="id", help="""type of conservation measure:"
     id - identity
     blos - blosum62
     jsdw - jensen shannon divergence with window (as in concavity)""")
    parser.add_argument("--alignment-test", default="", help="type of conservation measure, used for alignment. By default, like conservation_test")
    parser.add_argument("--coevolution-test", default="MI",
                        help="coevolution measure. By default, MI: mutual information. can be MIp, APC corrected MI")
    args = parser.parse_args()

    # CREATE LOG DETAILS
    if not os.path.exists("logs"):
        os.mkdir("logs")
    logging.basicConfig(level=logging.DEBUG, filename="logs/{}.txt".format(str(datetime.datetime.now()).replace(" ", "_")))
    logging.warning("LOGFILE INITIATED\nPIPELINE STARTED\nOk, I am back to life and ready to conquer the wo... oh.. damn..\n")

    # OUTPUT FOLDER CREATION
    o = args.output_path
    if not os.path.exists(o):
        os.makedirs(o)
        logging.warning("~~~\nCREATING OUTPUT FOLDER at {}\n~~~".format(o))

    # CONSERVATION check
    if not args.alignment_test:
        args.alignment_test = args.conservation_test

    # INPUT FILE READER
    read_input_file(args)

    # REPORT ARGS
    logging.warning("~~~~~pipeline has the following args:\n{}\n~~~~~".format(args))

    # CON-CON comparison
    if args.test_type == "CONCON":
        con_con_comparison(args)

    # DIRP
    if args.test_type == "DIRP":
        dir_prediction(args)

    if args.test_type == "COEVOL":
        coevol_test(args)

    if args.test_type == "LIGTAB":
        _ = ligand_table(args)

    # PCA from ligand table
    if args.test_type == "PCA":
        pca_test(args)

if __name__ == '__main__':
    main(sys.argv)

