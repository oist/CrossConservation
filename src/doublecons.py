# selection of classes for double conservation script

import re
from Bio import SeqIO
from Bio.SubsMat.MatrixInfo import blosum62
import pandas as pd
import numpy as np
from collections import Counter

class ConScores:
    """
    class containing conservation scores.
    They all input msa, reference.
    They all output an array based on ref positions:
    identity - returns frequency of ref residues in alignment
    """

    # general variab

    # blosum 62 marginal probabilities
    blos62_mp = {'A': 0.074,
                 'C': 0.025,
                 'D': 0.054,
                 'E': 0.054,
                 'F': 0.047,
                 'G': 0.074,
                 'H': 0.026,
                 'I': 0.068,
                 'K': 0.058,
                 'L': 0.099,
                 'M': 0.025,
                 'N': 0.045,
                 'P': 0.039,
                 'Q': 0.034,
                 'R': 0.052,
                 'S': 0.057,
                 'T': 0.051,
                 'V': 0.073,
                 'W': 0.013,
                 'Y': 0.032}

    # scoring methods

    @staticmethod
    def identity(msa, ref):
        """
        identity score: number of same aa as reference over the total number of aa on the position.
        :param msa:
        :param ref:
        :return:
        """
        out = []
        reflen = len(msa.loc[ref])

        for i in range(reflen):
            res_list = list(msa.iloc[:, i])
            reference_res = str(msa.iloc[:, i].loc[ref])
            ll = len(res_list) - 1  # cause we are not considering ref line
            this_freq = -1 / ll  # starts in debt to cover self reference hit

            if reference_res not in "-.":
                for res in res_list:
                    if res == reference_res:
                        this_freq += 1 / ll
                # print(", ".join([reference_res, "\t"] + res_list + [str(round(this_freq,2))]))
                out.append(this_freq)

        return out

    @staticmethod
    def blosum(msa, ref):
        """
        blosum score: sum of BLOSUM(ref, L) for L in other ligands.
        Normalized by the number of sums
        :param msa:
        :param ref:
        :return:
        """

        out = []
        reflen = len(msa.loc[ref])

        for i in range(reflen):
            norm = 0
            res_list = list(msa.iloc[:, i])
            reference_res = str(msa.iloc[:, i].loc[ref])

            if reference_res not in "-.":
                this_score = - blosum62[(reference_res, reference_res)]  # starts in debt to cover self reference hit
                for res in res_list:
                    if res not in "-.":
                        norm += 1
                        # exception to catch stupid triangular matrix
                        try:
                            bloscore = blosum62[(reference_res, res)]
                        except KeyError:
                            bloscore = blosum62[(res, reference_res)]
                        this_score += bloscore
                # print(", ".join([reference_res, "\t"] + res_list + [str(round(this_freq,2))]))
                out.append(np.round(this_score / norm, 1))

        return out

    @staticmethod
    def JSDw(msa, ref, w=1, lamb=1/2):
        """
        Jensen shannon divergence with window of residues.
        as in https://academic.oup.com/bioinformatics/article/23/15/1875/203579
        used by consurf!

        RE = sum aa Pc(aa) * log (Pc(aa) / q(aa))

        where Pc is the probability to find aa in the column C,
        q is the background probability of aa

        DJS is defined for a column C as
        Dc = lambda REpc,r + (1 - lambda) REq,r

        where r = lambda Pc + (1-lambda)q

        :param msa:
        :param ref:
        :param w:
        :param l:
        :return:
        """

        # safety checks
        assert w%2 == 1, "NOWAY, the window of residues must be odd, are you crazy!?"

        def JSD_SCORE(arr, lamb=1/2, gap_limit=1.1):
            """
            cons score for one column
            :param arr:
            :return:
            """

            # find aa freq
            count = Counter(arr)
            Pc = {it: c for it, c in count.items()}

            # banned aa
            if "X" in Pc:
                del(Pc["X"])

            # gap penalty measure
            l_pen = sum(Pc.values())
            gap_pen = 0
            if "-" in Pc:
                gap_pen = Pc["-"]
                del (Pc["-"])
            elif "." in Pc:
                gap_pen = Pc["."]
                del (Pc["."])
            gap_pen = gap_pen / l_pen
            # if gap are more than limit, return score 0
            if gap_pen > gap_limit:
                return 0.0

            # occurrences
            l = sum(Pc.values())
            # frequency
            for it in Pc:
                Pc[it] = Pc[it] / l

            # compute q and r dictionaries
            q = ConScores.blos62_mp  # dict of "aa" -> blos62 frequency
            r = {x: (lamb*Pc[x]) + ((1-lamb)*q[x]) for x in Pc}  # dict of "aa" -> r score

            # add aa to r and Pc if not present with (almost) zero
            # for k in q:
            #     if k not in r:
            #         r[k] = 10**-6
            #     if k not in Pc:
            #         Pc[k] = 10**-6

            # RE score with case for zero division inside log (it tends to 1)
            def RE(d1, d2):
                o = []
                for x in set(d1.keys()) & set(d2.keys()):
                    try:
                        v = d1[x]*(np.log(d1[x]/d2[x]))
                    except ZeroDivisionError:
                        v = d1[x]
                    o.append(v)
                return sum(o)

            # print(Pc)
            # print(sum(Pc.values()))
            # print(q)
            # print(sum(q.values()))
            # print(r)
            # print(sum(r.values()))
            # finally compute score
            DC = (lamb * RE(Pc, r)) + ((1 - lamb) * RE(q, r))
            # add gap penalty
            DC = DC * (1 - gap_pen)
            return DC

        #######################

        score_list = []
        reflen = len(msa.loc[ref])

        # compute JSD score for all ref positions
        for i in range(reflen):
            res_list = list(msa.iloc[:, i])
            reference_res = str(msa.iloc[:, i].loc[ref])

            if reference_res not in "-.":

                this_score = JSD_SCORE(res_list, lamb)
                score_list.append(this_score)


        # compute window score from score list
        out = []
        for i in range(len(score_list)):
            # window array init with same score
            wa = [score_list[i]]
            midpos = (w-1)//2 + 1

            # add positions surrounding middle positions
            for j in range(1, midpos):
                try:
                    wa.append(score_list[i + j])
                except IndexError:
                    pass
                try:
                    wa.append(score_list[i - j])
                except IndexError:
                    pass

            # now append the new score for position i in out list
            out.append(lamb*score_list[i] + (1-lamb)*np.mean(wa))

        # print(score_list)
        # print("\t".join(list(msa.loc[ref])))
        # print("\t".join([str(x) for x in out]))
        return out


    @staticmethod
    def weird_freq_test(msa, ref):
        # print(reslist)
        # print(ptrans(reslist,blosum62))
        # print(blosum62)
        # print("\n".join(["{} => {:.2f}".format(x[0], (2**x[1])* fd.get(x[0][0],10**-7)*fd.get(x[0][1],10**-7) ) for x in blosum62.items()]))
        # print(len(set([x[0] for x in blosum62.keys()])))
        allowed = set([x[0] for x in m.keys()])
        p = 1
        tot_len = len(l)
        l = [x for x in l if x in allowed]
        if len(l) == 0:
            return 0
        for res_outer in l:
            outer_prob = 0
            for res_inner in l:
                outer_prob += blosum62[(res_inner, res_outer)]
            outer_prob -= blosum62[(res_outer, res_outer)]  # because he considered himself once

    scoresdict = {
        "id": identity.__func__,
        "blos": blosum.__func__,
        "jsdw": JSDw.__func__
    }

class Protein(object):

    def __init__(self):

        # meta attrib
        #self._msa_cons_scores = None
        self.msa_cons_scores = None

        #self._prot_msa = None
        self.prot_msa = None

        # reference
        self.reference = ""
        # name
        self.name = ""

    # PROPERTY CLASSES

    @property
    def prot_msa(self):
        return self._prot_msa

    @prot_msa.setter
    def prot_msa(self, msa):
        # put here the feasibility tests
        self._prot_msa = msa

    @property
    def msa_cons_score(self):

        return self._msa_cons_scores

    @msa_cons_score.setter
    def msa_cons_score(self, arr):
        self._msa_cons_scores = arr

    # END OF PROPERTY CLASSES

    def get_cons_scores(self, score_func="id"):
        """special get function to calculate score if it has not been calculated"""

        if not self.msa_cons_scores:
            score_func = ConScores.scoresdict[score_func]  # fetch the specified conservation score type function
            scores = self.calculate_msa_cons_scores(score_func)
            self.msa_cons_scores = scores
        return self.msa_cons_scores

    def calculate_msa_cons_scores(self, score_func=ConScores.identity):
        """returns a score_array"""
        msa = self.prot_msa
        ref = self.reference

        assert ref in msa.index, "reference {} not found in MSA with indexes {}".format(ref, msa.index)
        mcs = score_func(msa, ref)

        return mcs

    def get_msa(self):
        return self.prot_msa

    def get_names(self):
        return self.prot_msa.index

    def get_ref_array(self):
        """
        returns an array of len(prot_msa) with positions and residue in respect of ref seq
        :return:
        """
        s = []  # reference residues
        p = []  # reference positions

        msalen = len(self.prot_msa.loc[self.reference])

        for i in range(msalen):
            reference_res = str(self.prot_msa.iloc[:, i].loc[self.reference])

            s.append(reference_res)
            if reference_res not in "-.":
                p.append(i)

        min_p = p[0]
        p = set(p)
        ref_count = 1
        out = []
        for i in range(-min_p, 0):  # before 0
            out.append("{}-".format(i))
        for i in range(0, msalen - min_p):  # after 0

            this_ref = s[min_p + i]  # reference in this position

            if this_ref in ".-":  # report inter-reference position
                ic += 1
                ref_str = "{}i{}".format(ref_count, ic)
            else:  # report reference position
                ic = 0
                ref_str = "{}".format(ref_count)
                ref_count += 1

            this_pos = "{}{}".format(ref_str, this_ref)

            out.append(this_pos)
        return out

    def get_referenced_pos(self):
        """
        returns referenced positions array, like in [1S, 2U, 3H, 4G...]
        :return:
        """
        ref_seq = self.get_ref_seq()
        ref_len = len(ref_seq)
        out = ["{}{}".format(i + 1, ref_seq[i]) for i in range(ref_len)]

        return out

    def get_ref_pos(self):
        """
        returns MSA position of non empty reference
        :return:
        """
        out = []
        reflen = len(self.prot_msa.loc[self.reference])
        for i in range(reflen):
            if list(self.prot_msa.iloc[:, i].loc[self.reference])[0] not in ".-":
                out.append(i)
        return out


    def get_ref_seq(self):
        """
        returns ref seq excluding gaps
        :return:
        """
        s = []

        reflen = len(self.prot_msa.loc[self.reference])

        for i in range(reflen):
            reference_res = str(self.prot_msa.iloc[:, i].loc[self.reference])

            if reference_res not in "-.":
                s.append(reference_res)
        return "".join(s)

    def get_seq(self, seqname):
        """
        returns seq excluding gaps
        :return:
        """
        s = []

        namelen = len(self.prot_msa.loc[seqname])

        for i in range(namelen):
            name_res = str(self.prot_msa.iloc[:, i].loc[seqname])

            if name_res not in "-.":
                s.append(name_res)
        return "".join(s)


class Ligand(Protein):
    """
    attribs:
    name        -   the gene name of the ligand
    description -   the description of msa type
    uprot       -   the uniprot ID of the ligand
    affinity    -   the degree of affinity (high,low)
    pdb         -   just the pdb id
    dna_seq     -   the dna seq
    prot_seq    -   the prot seq
    reference   -   the reference organism for MSA

    meta attribs:
    msa_cons_scores     -   the msa cons score
    prot_msa            -   the protein msa
    dna_msa             -   same but dna

    ####
    MSA
    msa is a pandas dataframe.

    ###
    Functions:

    USE get_cons_scores TO GET THE SCORES!

    """
    def __init__(self, label, uprot="", affinity="", pdb="", prot_seq="", reference="Hsap", description=""):

        super(Ligand,self).__init__()
        # inputs
        self.name = label
        self.uprot = uprot
        self.affinity = affinity
        self.pdb = pdb
        self.prot_seq = prot_seq
        self.reference = reference


class Receptor(Protein):
    """
    attribs:
    name        -   the gene name of the ligand
    description -   the description of msa type
    uprot       -   the uniprot ID of the ligand
    affinity    -   the degree of affinity (high,low)
    pdb         -   just the pdb id
    dna_seq     -   the dna seq
    prot_seq    -   the prot seq
    reference   -   the reference organism for MSA

    meta attribs:
    msa_cons_scores     -   the msa cons score
    prot_msa            -   the protein msa
    dna_msa             -   same but dna

    ####
    MSA
    msa is a pandas dataframe.

    ###
    Functions:

    USE get_cons_scores TO GET THE SCORES!

    """
    def __init__(self, label, uprot="", affinity="", pdb="", prot_seq="", reference="Hsap", description=""):

        super(Receptor, self).__init__()
        # inputs
        self.name = label
        self.uprot = uprot
        self.affinity = affinity
        self.pdb = pdb
        self.prot_seq = prot_seq
        self.reference = reference


class MSA(object):
    """
    class representing a multiple sequence/structure alignment, with useful functions

    attribs:
    name        -   the gene name of the ligand
    description -   the description of msa type
    reference   -   the reference sequence

    """
    def __init__(self, name="", description="", reference=""):

        self.name = name
        self.description = description
        self.reference = reference

        # meta attrib
        self.msa_cons_scores = None

        self.msa = None

        # PROPERTY CLASSES

        @property
        def msa(self):
            return self._msa

        @msa.setter
        def msa(self, msa):
            # put here the feasibility tests
            self._msa = msa

        @property
        def msa_cons_score(self):
            return self._msa_cons_scores

        @msa_cons_score.setter
        def msa_cons_score(self, arr):
            self._msa_cons_scores = arr

    def get_msa(self):
        return self.msa

    def get_ref_pos(self):
        """
        returns MSA position of non empty reference
        :return:
        """
        out = []
        reflen = len(self.msa.loc[self.reference])
        for i in range(reflen):
            if list(self.msa.iloc[:, i].loc[self.reference])[0] not in ".-":
                out.append(i)
        return out

    def get_cons_scores(self, score_func="id"):
        """special get function to calculate score if it has not been calculated"""
        if not self.msa_cons_scores:
            score_func = ConScores.scoresdict[score_func]  # fetch the specified conservation score type function
            scores = self.calculate_msa_cons_scores(score_func)
            self.msa_cons_scores = scores
        return self.msa_cons_scores

    def calculate_msa_cons_scores(self, score_func=ConScores.identity):
        """returns a score_array"""
        msa = self.msa
        ref = self.reference

        assert ref in msa.index, "reference {} not found in MSA with indexes {}".format(ref, msa.index)
        mcs = score_func(msa, ref)

        return mcs

    def get_ref_seq(self):
        """
        returns ref seq excluding gaps
        :return:
        """
        s = []

        reflen = len(self.msa.loc[self.reference])

        for i in range(reflen):
            reference_res = str(self.msa.iloc[:, i].loc[self.reference])

            if reference_res not in "-.":
                s.append(reference_res)
        return "".join(s)

    def get_names(self):
        return self.msa.index

    def get_referenced_pos(self):
        """
        returns referenced positions array, like in [1S, 2U, 3H, 4G...]
        :return:
        """
        ref_seq = self.get_ref_seq()
        ref_len = len(ref_seq)
        out = ["{}{}".format(i + 1, ref_seq[i]) for i in range(ref_len)]

        return out

    def get_position(self, name, idx):
        """
        returns the residues for sequence "name" in reference position "idx".
        The last residue is the match, "-" if there is no match.
        :param name:
        :param idx:
        :return:
        """
        msa = self.msa
        ref = self.reference
        assert name in msa.index, "{} not found in the MSA with names {}".format(name, msa)
        assert ref in msa.index, "ref {} not found in the MSA with names {}".format(name, msa)

        refidx = 0
        namepos = "None"
        current_pos = []
        reflist = msa.loc[ref]
        namelist = msa.loc[name]

        reflen = len(reflist)
        assert 0 < idx <= reflen, "given idx is impossible!"

        for i in range(reflen):
            if namelist[i] not in ".-":  # by default dont add gaps
                current_pos.append(namelist[i])

            if refidx + 1 == idx:
                if namelist[i] in ".-":  # if the last one is a gap, add it
                    current_pos.append(namelist[i])
                namepos = ",".join(current_pos)
                break

            if reflist[i] not in ".-":
                refidx += 1
                current_pos = []

        return namepos

    def get_name_seq(self, name):
        """
                returns name seq excluding gaps
                :return:
                """
        s = []

        namelen = len(self.msa.loc[name])

        for i in range(namelen):
            name_res = str(self.msa.iloc[:, i].loc[name])

            if name_res not in "-.":
                s.append(name_res)
        return "".join(s)

    @staticmethod
    def pseq_phase(pseq, mseq):
        """
        return the phase and the cm where
        :param pseq:
        :param mseq:
        :return:
        """
        best_couple = []
        counter = 0

        limcheck = 0.9 * len(mseq)  # if 60% is aligned, then consider it right
        cm = -1

        while cm < limcheck:
            cm += 1
            phase = -1
            while phase < limcheck:
                best_couple.append((counter,(cm,phase-cm)))
                c = -1
                phase += 1
                counter = 0
                while c < limcheck and len(mseq) > cm + c + 1 and len(pseq) > c + phase + 1:
                    c += 1
                    if mseq[cm + c] == pseq[c + phase]:
                        counter += 1
                        if counter >= limcheck - 0.1:
                            return cm, phase - cm
                        # print(counter)

        cm, phase = sorted(best_couple)[-1][1]
        return cm,phase
        #
        # raise ValueError("can't combine these two sequences:\n{}\n{}".format(pseq,mseq))

    def msa_to_evo_reference(self, prot):
        """
        returns a dictionary of key: prot (reference) position, values: msa (name) position
        STARTS FROM 1
        :param prot:
        :return:
        """
        out = {}
        name = prot.name
        # print(name)
        pseq = prot.get_ref_seq()
        mseq = self.get_name_seq(name)
        # print(mseq)
        # print(pseq)
        
        cm, phase = self.pseq_phase(pseq,mseq)
        # print(cm,phase)
        for i in range(len(mseq) + phase + cm):
            if i < cm:
                out[i+1] = "na"
            else:
                out[i+1] = i + phase + 1

        return out


    def get_referenced_positions(self, name):
        """
        returns a list of positions split as to be re
        :param name:
        :param ref:
        :return:
        """

        msa = self.msa
        ref = self.reference
        assert name in msa.index, "{} not found in the MSA with names {}".format(name, msa.index)
        assert ref in msa.index, "ref {} not found in the MSA with names {}".format(ref, msa.index)

        name_out = []
        current_pos = []
        reflist = list(msa.loc[ref])
        namelist = list(msa.loc[name])
        namepos = 0
        reflen = len(reflist)
        last_ref_pos = 0

        for i in range(reflen):
            if namelist[i] not in ".-":  # by default dont add gaps
                namepos += 1
                current_pos.append("{}{}".format(namepos,namelist[i]))

            if reflist[i] not in ".-":
                last_ref_pos = i + 0
                if namelist[i] in ".-":  # if the last one is a gap, add it
                    current_pos.append("-")
                name_out.append(",".join(current_pos))
                current_pos = []

        # add tail of extra residues
        if current_pos:
            # add a gap if there is a gap in last position
            if namelist[last_ref_pos] in "-.":
                current_pos.insert(0, "-")
            else:
                current_pos.insert(0,name_out[-1])  # add last position added if there was a match

            name_out[-1] = ",".join(current_pos)

        return name_out

    def get_referenced_scores(self, name, prot, score_arr):
        """
        returns a list of conservation scores ordered as in reference
        This is used in reference ordered table print
        There are two position to be translated.
        First, the translation between MSA reference and MSA name are found with self.get_referenced_position
        Then a mapping of MSA name to EVOLUTIONARY name is done with self.msa_to_evo_reference function
        This is done because the EVO score_array is based on EVOLUTIONARY reference.
        Following that, ref_pos are split and converted to integers to be used inside msa2evo, that are the indexes of score_array
        easy
        :param name:
        :param score_arr: score array
        :return:
        """

        ref_pos = self.get_referenced_positions(name)  # msa positions in reference len array

        msa2evo = self.msa_to_evo_reference(prot)
        # print(msa2evo)
        # print(len(score_arr))
        # print(score_arr)
        # print(ref_pos[0], ref_pos[-1])

        # initialize ref length array
        out = ["-"] * len(ref_pos)
        i = 0
        try:
            for pos_string in ref_pos[:-1]:
                pos_string = pos_string.split(",")[-1]  # the last in comma sep values ex. 5E,6Y,7P
                if pos_string != "-":
                    this_pos = int(pos_string[:-1])  # excluding residue, ex. 5E
                    converted_pos = msa2evo[this_pos]
                    if converted_pos != "na":
                        # print(this_pos, pos_string, round(score_arr[converted_pos - 1],2))
                        out[i] = "{}".format(round(score_arr[converted_pos - 1],2))
                i += 1

            # in case of last residue, the aligned aa is the first, not the last!
            last_score = ref_pos[-1].split(",")[0]
            if last_score != "-":
                this_pos_name = int(last_score[:-1])
                converted_pos = msa2evo[this_pos_name]
                if converted_pos != "na":
                    out[-1] = "{}".format(round(score_arr[converted_pos - 1],2))
        except IndexError:
            pass
        return out

def create_msa(f):
    """created a msa pandas dataframe from an open file f"""
    prsr = SeqIO.parse(f, "fasta")
    df = pd.DataFrame()
    for fas in prsr:
        df = df.append(pd.Series(list(fas.seq), index=range(1, len(fas.seq) + 1), name=fas.id))
    return df

def mutual_info(a):
    """ from array of coupled positions, returns MI score:
    sum x, sum y, of Px,y * (log Px,y over Px*Py)
    """
    mi = 0.0
    N = len(a)
    if N <= 1:
        return mi

    p1 = Counter([x[0] for x in a])  # occurrence count in pos1
    p2 = Counter([x[1] for x in a])  # occurrence count in pos2
    coup = Counter(a)
    for item, c in coup.items():  # occurrence count of couple
        # print(np.log((c/(p1[item[0]]*p2[item[1]]))))
        # print(np.log(N))
        mi += (c / N) * (np.log10(((c/N) / ((p1[item[0]]/N) * (p2[item[1]]/N)))))

    return mi

def intra_coevol(prot, outpath):
    """
    intra protein coevolution
    :param msa:
    :param org:
    :param out:
    :return:
    """

    ref_pos = prot.get_ref_pos()
    referenced_pos = prot.get_referenced_pos()
    ref_len = len(ref_pos)
    msa = prot.get_msa()
    org_list = prot.get_names()

    miatrix = np.zeros([ref_len, ref_len])
    top5list = []

    for i in range(ref_len - 1):
        ipos = ref_pos[i]
        icol = msa.iloc[:, ipos]
        for j in range(i + 1, ref_len):
            test = []
            jpos = ref_pos[j]
            jcol = msa.iloc[:, jpos]

            # find couples
            for org in org_list:
                org_ires = list(icol[org])[0]
                org_jres = list(jcol[org])[0]
                if org_ires not in ".-" and org_jres not in ".-":
                    test.append((org_ires,org_jres))
            MI = mutual_info(test)
            MI = round(MI, 2)
            miatrix[i][j] = MI
            top5list.append((MI, referenced_pos[i], referenced_pos[j]))

    mitab = pd.DataFrame(miatrix, columns=referenced_pos, index=referenced_pos)

    top5tab = pd.DataFrame(sorted(top5list, reverse=True)[:int(len(top5list) * 0.10)])

    with open(outpath + ".tsv", "w") as outfile:
        mitab.to_csv(outfile, sep="\t")

    with open(outpath + "_top5.tsv", "w") as outfile:
        top5tab.to_csv(outfile, sep="\t")

def inter_coevol(p1, p2, outpath):
    """
    intra protein coevolution
    :param msa:
    :param org:
    :param out:
    :return:
    """

    r1_pos = p1.get_ref_pos()
    r2_pos = p2.get_ref_pos()
    referenced1_pos = p1.get_referenced_pos()
    referenced2_pos = p2.get_referenced_pos()
    r1_len = len(r1_pos)
    r2_len = len(r2_pos)
    msa1 = p1.get_msa()
    msa2 = p2.get_msa()
    org_set = set(p1.get_names()) & set(p2.get_names())

    miatrix = np.zeros([r1_len, r2_len])
    top5list = []

    for i in range(r1_len):
        ipos = r1_pos[i]
        icol = msa1.iloc[:, ipos]
        for j in range(r2_len):
            test = []
            jpos = r2_pos[j]
            jcol = msa2.iloc[:, jpos]

            # find couples
            for org in org_set:
                org_ires = list(icol[org])[0]
                org_jres = list(jcol[org])[0]
                if org_ires not in ".-" and org_jres not in ".-":
                    test.append((org_ires,org_jres))
            MI = mutual_info(test)
            MI = round(MI, 2)
            miatrix[i][j] = MI
            top5list.append((MI, referenced1_pos[i], referenced2_pos[j]))

    mitab = pd.DataFrame(miatrix, columns=referenced2_pos, index=referenced1_pos)

    top5tab = pd.DataFrame(sorted(top5list, reverse=True)[:int(len(top5list) * 0.10)])

    with open(outpath + ".tsv", "w") as outfile:
        mitab.to_csv(outfile, sep="\t")

    with open(outpath + "_top5.tsv", "w") as outfile:
        top5tab.to_csv(outfile, sep="\t")