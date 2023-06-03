from itertools import product
import pandas as pd
import re
from bs4 import BeautifulSoup as bs
import markov

def get_usage_df():
    """
    Returns a pandas dataframe with columns 'triplet', 'amino acid' and 'number' 
    loaded from Codon_usage_table.html
    """

    with open("src/codon/Codon_usage_table.html") as file:
        soup = bs(file, "html.parser")
    pre = soup.select_one("pre").text

    rows = []
    pattern = r"([AUCG]{3,3})\s([A-Z\*])\s\d.\d\d\s+\d{1,2}.\d\s\(\s*(\d+)\)"
    for line in pre.split("\n"):
        lst = re.findall(pattern, line)
        
        for group in lst:
            triplet, amino_acid, n = group
            s = pd.Series({"triplet" : triplet, "amino acid" : amino_acid, "number" : int(n)})
            rows.append(s)
    return pd.DataFrame(rows)

def get_codon_to_amino_dict():
    "Returns a dictionary with triplets as keys and the mapped amino acid as values."

    df = get_usage_df()
    return dict(zip(df["triplet"], df["amino acid"]))

def get_amino_to_codon_dict():
    """
    Returns a dictionary with amino acids as keys and a list of triplets coding 
    the amino acid as values.
    """

    df = get_usage_df()
    grouped_df = df.groupby("amino acid")["triplet"]

    return { key : list(grouped_df.get_group(key).values) for key, _ in grouped_df }

def get_codon_probability_dict():
    """
    Returns a dictionary with triplets as keys and the probability of the 
    corresponding amino acid being coded from this triplet as values.
    """

    df = get_usage_df()

    amino_acids_total = pd.DataFrame(df.groupby("amino acid")["number"].sum())
    amino_acids_total.columns = ["amino acid number"]

    merged_df = pd.merge(df, amino_acids_total, on="amino acid")
    return dict(
        zip(
            merged_df["triplet"], 
            merged_df["number"].astype(float) / merged_df["amino acid number"]
        )
    )

def codon_probabilities(rna):
    """
    Splits the given RNA sequnce 'rna' in to a list 'L' of 3-mers. Then for every 
    3-mer W in the list L, calculates the fraction 'total number of W in the list L' 
    divided by 'the total number 3-mers in the list L mapping to the same amino acid as W'. 

    Returns the resulting dictionary with 3-mers as keys and the fraction as values.
    Contains a mapping for every possible 3-mer.
    """

    k = 3
    codon_counts = dict()
    kmer_indexes = markov.kmer_index(rna, k)
    # first calculate the number of appearences for each possible 3-mer
    for codon_lst in product("ACGU", repeat = k):
        codon = "".join(codon_lst)
        if codon in kmer_indexes:
            codon_counts[codon] = len(list(filter(lambda i: (i - k) % k == 0, kmer_indexes[codon])))
        else:
            codon_counts[codon] = 0

    codon_probs = dict()
    amino_to_codon = get_amino_to_codon_dict()
    # lastly, calculate the fractions
    for codon_list in amino_to_codon.values():
        total_codons = 0
        for codon in codon_list:
            total_codons += codon_counts[codon]

        for codon in codon_list:
            if total_codons == 0:
                codon_probs[codon] = 0.0
            else:
                codon_probs[codon] = float(codon_counts[codon]) / total_codons
        
    return codon_probs