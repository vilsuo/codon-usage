import codon.codon_usage as codon_usage
from markov import random_event

def dna_to_rna(s):
    "returns DNA s converted into RNA"

    encoding_rules = dict(zip("ACGT","ACGU"))
    return "".join(encoding_rules[nucleon] for nucleon in s)

def rna_to_protein(s):
    "returns RNA s converted into protein. Assumes that the length of s is a multiple of three"

    n = 3
    triplets = [s[i : i + n] for i in range(0, len(s), n)]
    conversion_rules = codon_usage.get_codon_to_amino_dict()
    
    return "".join([conversion_rules[triplet] for triplet in triplets])

def dna_to_protein(s):
    return rna_to_protein(dna_to_rna(s))

class ProteinToRNA:
    
    def __init__(self):
        self.codon_to_amino_probability_dict = codon_usage.get_codon_probability_dict()
        self.amino_to_codon_list_dict = codon_usage.get_amino_to_codon_dict()

    def convertMax(self, s):
        "return a RNA which is most likely the source of the protein s"

        lst = []
        for amino in s:
            codon_lst = self.amino_to_codon_list_dict[amino]

            most_likely_prob = 0.0
            for codon in codon_lst:
                codon_prob = self.codon_to_amino_probability_dict[codon]
                if codon_prob > most_likely_prob:
                    most_likely_codon = codon
                    most_likely_prob = codon_prob
        
            lst.append(most_likely_codon)
        return "".join(lst)
    
    def convertRandom(self, s):
        "return a random RNA which is a source of protein s"

        lst = []
        for amino in s:
            codon_list = self.amino_to_codon_list_dict[amino]

            dist = dict(zip(codon_list, [self.codon_to_amino_probability_dict[codon] for codon in codon_list]))
            lst.append(random_event(dist))

        return "".join(lst)