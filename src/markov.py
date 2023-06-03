from functools import reduce
import random
import numpy as np
from itertools import product

def random_event(dist, seed = None):
    "Returns a randomly selected event from the distribution 'dist'."

    random.seed(seed)
    x = random.uniform(0, 1)
    cumsum = 0.0
    for key, value in dist.items():
        cumsum += value
        if x < cumsum:
            return key
        
"""def sliding_window(s, k):
    
    This function returns a generator that can be iterated over all starting position of 
    a k-window in the sequence. For each starting position the generator returns the 
    nucleotide frequencies in the window as a dictionary.
    
    freqs = {"A" : 0, "C" : 0, "G" : 0, "T" : 0}
    for i in range(k):
        new = s[i]
        freqs[new] += 1

    yield freqs

    i = 1
    while (i + k - 1) < len(s):
        old = s[i - 1]
        new = s[i + k - 1]
        
        freqs[old] -= 1
        freqs[new] += 1

        i += 1
        yield freqs"""

def get_context_frequncies(s, k):
    """
    Return dictionary associating for each k-mer 'W' of 's' a dictionary containing the 
    frequencies of alphabets appearing right after 'W', or equivalently: 
    
        { W : {c : frequency of substrings Wc of s } for each k-mer W of s }.

    Does not contain mappings for missing k-mers/missing alphabets for existing k-mers.
    """

    context_freqs = dict()
    for i in range(k, len(s)):
        k_mer = s[(i - k) : i]
        c = s[i]

        if k_mer not in context_freqs:
            context_freqs[k_mer] = dict()
        
        if c not in context_freqs[k_mer]:
            context_freqs[k_mer][c] = 0

        context_freqs[k_mer][c] += 1

    return context_freqs

def get_context_probabilities(s, k):
    """
    Returns a dictionary associating for each k-mer 'W' of 's' a dictionary containing 
    the probabilities of alphabets appearing right after 'W', or equivalently: 
    
        { W : {c : probability of c appearing right after W} for each k-mer of s }.

    Does not contain mappings for missing k-mers/missing alphabets for existing k-mers.
    """

    return {
        context : {
            alphabet : alphabet_frequencies[alphabet] / np.sum(list(alphabet_frequencies.values()))
            for alphabet in alphabet_frequencies.keys()
        }
        for context, alphabet_frequencies in get_context_frequncies(s, k).items()
    }

def get_context_pseudo_probabilities(s, k):
    """
    Returns a dictionary associating for each length 'k' concatenation 'W' of alphabets 
    of 's' a dictionary containing the pseudo probabilities of alphabets appearing right 
    after 'W', or equivalently: 
    
        { W : {c : pseudo probability of c appearing right after W} for each length 'k' concatenation 'W' of alphabets of 's' }. 
        
    Pseudo probability means that the frequency of each alphabet initialized to the count 1.
    Does also contain mappings for each missing k-mers/missing alphabets for existing k-mers.
    """

    s_alphabet = set(s)
    context_freqs = get_context_frequncies(s, k)

    for pseudo_context_list in product(s_alphabet, repeat = k):
        pseudo_context = "".join(pseudo_context_list)

        # for every possible context in s, increment the frequencies of each alphabet in s and add if it does not exist
        if pseudo_context in context_freqs:
            context_freqs[pseudo_context] = {
                alphabet : (context_freqs[pseudo_context][alphabet] + 1 if alphabet in context_freqs[pseudo_context] else 1)
                for alphabet in s_alphabet
            }
        else:
            context_freqs[pseudo_context] = dict(zip(s_alphabet, np.full(len(s_alphabet), 1)))

    # calculate and return the probabilities
    return {
        context : { 
            alphabet : alphabet_frequencies[alphabet] / np.sum(list(alphabet_frequencies.values())) 
            for alphabet in alphabet_frequencies.keys() 
        }
        for context, alphabet_frequencies in context_freqs.items()
    }

def kmer_index(s, k):
    "Returns a dictionary containing the list if starting indices for each k-mer of s"

    kmer_index_list = dict()
    for i in range(len(s) - (k - 1)):
        kmer = s[i : i + k]
        if kmer not in kmer_index_list:
            kmer_index_list[kmer] = []
        kmer_index_list[kmer].append(i)
    return kmer_index_list

class MarkovChain:
    
    def __init__(self, zeroth, kth, k = 2):
        self.k = k
        self.zeroth = zeroth
        self.kth = kth
        
    def generate(self, n, seed = None):
        """
        Returns a randomly generated string sequence of length n, where the first self.k 
        values are sampled from distribution 'zeroth', and the remaining values are sampled 
        from distribution 'kth'.
        """
        
        nucleotides = []
        for i in range(n):
            if i < self.k:
                nucleotide = random_event(self.zeroth, seed)
            else:
                k_mer = "".join(nucleotides[-self.k:])
                nucleotide = random_event(self.kth[k_mer], seed)
            nucleotides.append(nucleotide)

        return "".join(nucleotides)
    
    def log_probability(self, s):
        """
        Returns the log2-probability that given distributions 'zeroth' and 'kth' the k-th order 
        Markov chain generates the string sequence 's'.
        """

        if len(s) == 1:
            return np.log2(self.zeroth[s[0]])

        probability = 0
        for i in range(len(s)):
            current = s[i]
            if i < self.k:
                # zeroth
                probability += np.log2(self.zeroth[current])
            else:
                # kth
                k_mer = s[i - self.k : i]
                probability += np.log2(self.kth[k_mer][current])
        return probability
    
def kullback_leibler(p, q):
    """
    Computes Kullback-Leibler divergence between two distributions.
    Both p and q must be dictionaries from events to probabilities.
    The divergence is defined only when q[event] == 0 implies p[event] == 0.
    """

    distance = 0
    for key in p.keys():
        if p[key] != 0:
            distance += p[key] * np.log2(float(p[key])/q[key])
    return distance

def get_stationary_distributions(transition):
    """
    The function gets a transition matrix of a degree one Markov chain as parameter.
    Returns a list of stationary distributions, in vector form, for that chain.
    """

    eigen_values, eigen_vectors = np.linalg.eig(transition.T)

    stationary_distributions = []
    for i, eigen_value in enumerate(eigen_values):
        if abs(eigen_value - 1.0) < 10**-10:
            eigen_vector = eigen_vectors[:, i]
            normalized = np.abs(eigen_vector) / np.sum(np.abs(eigen_vector))
            stationary_distributions.append(normalized)

    return stationary_distributions